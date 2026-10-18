"""Meaningful backend contracts. No training, inference, GPU worker or CVAT request."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import yaml
from annoloop.compatibility import checkpoint_metadata, matrix, relation
from annoloop.cvat import binary_pcd, label_specs, package, payload, verify_readback
from annoloop.fusion import calibrate_source, compare, normalize, source_calibration_report, SourceCalibrationError
from annoloop.inventory import fixed_samples, inspect_dataset, source_counts
from annoloop.io import digest, read, write
from annoloop.pretrain import load_config, source_config
from annoloop.projects import ProjectStore
from annoloop.schema import ClassSchema, discover, from_coco_source
from annoloop.selection import SelectionPolicy, decide
from annoloop.tracking import refine
from annoloop.source_worker import actual_source_classes, selected_source_inventory
from types import SimpleNamespace


def box(name="car", identity="a", x=4., score=.8):
    return dict(id=identity, class_name=name, box=[x, 0., 0., 4., 2., 2., 0.], score=score,
                confidence=.8, quality=.8)


def metrics(count=1, agreement=.9):
    return dict(domain="target_unlabelled_proxy", target_gold_read=False,
                agreement=agreement, supported_fraction=.9, box_count=count)


class BackendTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.schema = ClassSchema("3d", "cuboid", ("car", "truck"))
        write(self.dataset / "class_schema.json", self.schema.as_dict())
        np.save(self.dataset / "f0.npy", np.ones((4, 3), dtype=np.float32))
        self.store = ProjectStore(self.root / "storage")

    def project(self, policy=None):
        return self.store.create("contract fixture", self.dataset, policy=policy)

    def test_allow_continue_only_clears_finish_request(self):
        pid=self.project()
        self.store.request_finish(pid)
        before=self.store.get(pid)
        self.assertIsNone(self.store.next_round(pid))
        with patch('annoloop.adaptation.active_job',return_value=None), \
             patch('annoloop.adaptation.start_job') as start:
            after=self.store.allow_continue(pid)
            self.assertEqual(after,dict(before,stop_after_round=False))
            self.assertEqual(self.store.next_round(pid),1)
            self.assertFalse((self.store.path(pid)/'adaptation_jobs').exists())
            start.assert_not_called()

    def test_allow_continue_rejects_live_job_and_exhausted_budget(self):
        pid=self.project();self.store.request_finish(pid)
        before=self.store.get(pid)
        with patch('annoloop.adaptation.active_job',return_value='live job'):
            with self.assertRaisesRegex(ValueError,'Chờ job'):
                self.store.allow_continue(pid)
        self.assertEqual(self.store.get(pid),before)
        self.store.configure(pid,before['name'],0,'manual')
        with patch('annoloop.adaptation.active_job',return_value=None):
            with self.assertRaisesRegex(ValueError,'Đã hết số vòng'):
                self.store.allow_continue(pid)
        self.assertTrue(self.store.get(pid)['stop_after_round'])

    def record(self, project, index=1, agreement=.9):
        model = self.root / "model_fixture.pth"
        model.write_bytes(b'fixture only, never a real checkpoint')
        predictions = dict(target_gold_read=False, frames=[dict(frame_id="f0", boxes=[box()])])
        return self.store.record_round(project, index, predictions, metrics(agreement=agreement),
                                       {"A": model, "B": model}, {"f0": dict(boxes=[box()])})

    def test_schema_from_dataset_is_not_fixed_car_list(self):
        write(self.dataset / "class_schema.json", dict(data_type="3d", annotation_type="cuboid", classes=["forklift"]))
        self.assertEqual(discover(self.dataset)[0].classes, ("forklift",))

    def test_raw_cloud_needs_schema(self):
        (self.dataset / "class_schema.json").unlink()
        with self.assertRaisesRegex(ValueError, "schema"):
            inspect_dataset(self.dataset)

    def test_target_annotation_instances_are_never_opened(self):
        (self.dataset / "annotations.json").write_text("must not parse target gold")
        self.assertFalse(inspect_dataset(self.dataset)["target_gold_read"])

    def test_schema_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            ClassSchema.parse(dict(data_type="3d", annotation_type="cuboid", classes=["car", "car"]))

    def test_manifest_rejects_external_media(self):
        np.save(self.root / "outside.npy", np.ones((2, 3)))
        write(self.dataset / "manifest.json", dict(frames=[dict(frame_id="a", points="../outside.npy")]))
        with self.assertRaisesRegex(ValueError, "ngoài"):
            inspect_dataset(self.dataset)

    def test_sequence_cannot_leak_into_holdout(self):
        write(self.dataset / "manifest.json", dict(frames=[dict(frame_id="a", points="f0.npy", sequence_id="s", split="adaptation"),
            dict(frame_id="b", points="f0.npy", sequence_id="s", split="holdout")]))
        with self.assertRaisesRegex(ValueError, "sequence"):
            inspect_dataset(self.dataset)

    def test_fixed_samples_exclude_holdout_and_include_difficulty(self):
        inv = dict(frames=[dict(frame_id="h", split="holdout", difficulty="hard", sequence_id="s"),
                          dict(frame_id="a", split="adaptation", difficulty="easy", sequence_id="a"),
                          dict(frame_id="b", split="adaptation", difficulty="hard", sequence_id="b")])
        self.assertEqual(fixed_samples(inv, 2), ["a", "b"])

    def test_source_requires_explicit_role_and_split(self):
        path = self.root / "labels.json"
        write(path, dict(role="target", split="train", frames=[]))
        with self.assertRaisesRegex(ValueError, "source"):
            source_counts(path, self.schema)

    def test_source_missing_class_count_is_zero(self):
        path = self.root / "labels.json"
        write(path, dict(role="source", split="train", frames=[dict(boxes=[box()])]))
        self.assertEqual(source_counts(path, self.schema)["counts"], {"car": 1, "truck": 0})

    def test_source_worker_accepts_declared_extra_classes_when_training_subset(self):
        schema = ClassSchema("3d", "cuboid", ("car",))
        write(self.root / "class_schema.json", self.schema.as_dict())
        path = self.root / "source.json"
        write(path, dict(role="source", split="train", frames=[dict(boxes=[box(), box("truck")])]))
        result = selected_source_inventory(path, schema, "train")
        self.assertEqual(result["selected_counts"], {"car": 1})
        self.assertEqual(result["counts"], {"car": 1, "truck": 1})

    def test_source_worker_still_rejects_missing_selected_class_and_wrong_split(self):
        write(self.root / "class_schema.json", self.schema.as_dict())
        path = self.root / "source.json"
        write(path, dict(role="source", split="train", frames=[dict(boxes=[box()])]))
        with self.assertRaises(ValueError):
            selected_source_inventory(path, self.schema, "train")
        with self.assertRaises(ValueError):
            selected_source_inventory(path, ClassSchema("3d", "cuboid", ("car",)), "validation")

    def test_actual_source_infos_cannot_be_replaced_by_claimed_class_list(self):
        dataset = SimpleNamespace(infos=[dict(gt_names=["Vehicle"])])
        with self.assertRaisesRegex(ValueError, "thực thiếu"):
            actual_source_classes(dataset, ["car", "truck"])
        dataset = SimpleNamespace(infos=[dict(gt_names=["car", "truck", "car"])])
        self.assertEqual(actual_source_classes(dataset, ["car", "truck"]), {"car": 2, "truck": 1})

    def test_coco_category_names_are_read_in_id_order(self):
        path = self.root / "coco.json"
        write(path, dict(categories=[dict(id=2, name="person"), dict(id=1, name="car")]))
        self.assertEqual(from_coco_source(path).classes, ("car", "person"))

    def test_broad_vehicle_is_not_usable_as_car(self):
        rules = [dict(namespace="waymo:v1", target="car", native="Vehicle", relation="broader", verified=True)]
        self.assertFalse(relation("car", ["Vehicle"], "waymo:v1", rules)["usable"])
        self.assertEqual(relation("car", ["Vehicle"], "other:v1", rules)["status"], "missing")

    def test_unverified_alias_is_not_automatically_used(self):
        rule = dict(namespace="d:v1", target="pedestrian", native="person", relation="exact", verified=False)
        self.assertFalse(relation("pedestrian", ["person"], "d:v1", [rule])["usable"])

    def test_checkpoint_hash_and_verified_classes_are_required(self):
        checkpoint = self.root / "weights.pth"
        checkpoint.write_bytes(b'fixture')
        path = self.root / "meta.json"
        write(path, dict(checkpoint="weights.pth", checkpoint_sha256=digest(checkpoint), classes=["car"], classes_verified=True))
        self.assertEqual(checkpoint_metadata(path)["classes"], ["car"])
        checkpoint.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            checkpoint_metadata(path)

    def test_source_presence_does_not_imply_checkpoint_support(self):
        sources = [dict(id="s", namespace="s:v1", classes=["car", "truck"], train_counts={"car": 10, "truck": 10}, validation_counts={"car": 1})]
        models = [dict(id="m", namespace="m:v1", classes=["car"], classes_verified=True)]
        row = matrix(self.schema, sources, models)[1]
        self.assertTrue(row["sources"]["s"]["train_ready"])
        self.assertFalse(row["sources"]["s"]["calibration_ready"])
        self.assertFalse(row["models"]["m"]["usable"])

    def test_manual_review_blocks_next_round(self):
        project = self.project()
        self.assertEqual(self.record(project)["action"], "review")
        with self.assertRaises(ValueError):
            self.store.next_round(project)
        self.store.select(project, "M1")
        self.assertEqual(self.store.next_round(project), 2)

    def test_retains_snapshot_and_checkpoint_after_original_changes(self):
        project = self.project()
        self.record(project)
        (self.root / "model_fixture.pth").unlink()
        self.store.select(project, "M1")
        with self.assertRaises(ValueError):
            self.record(project, 1)

    def test_auto_rejection_preserves_previous_selected_version(self):
        project = self.project(SelectionPolicy(mode="automatic"))
        self.record(project)
        decision = self.record(project, 2, agreement=.5)
        self.assertEqual(decision["action"], "reject")
        self.assertEqual(self.store.get(project)["selected_version"], "M1")

    def test_stop_allows_active_round_to_finish_but_no_next_round(self):
        project = self.project()
        self.store.request_finish(project)
        self.record(project)
        self.assertIsNone(self.store.next_round(project))

    def test_input_mutation_rejected_before_round_snapshot(self):
        project = self.project()
        np.save(self.dataset / "f0.npy", np.zeros((4, 3)))
        with self.assertRaisesRegex(ValueError, "Media"):
            self.record(project)

    def test_changed_snapshot_cannot_be_selected(self):
        project = self.project()
        self.record(project)
        write(self.store.path(project) / "rounds/M1/labels.json", {})
        with self.assertRaisesRegex(ValueError, "Snapshot"):
            self.store.select(project, "M1")

    def test_round_requires_fixed_sample_predictions(self):
        project = self.project()
        with self.assertRaisesRegex(ValueError, "Sample"):
            self.store.record_round(project, 1, dict(target_gold_read=False, frames=[dict(frame_id="f0", boxes=[box()])]),
                                     metrics(), {}, {})

    def test_policy_does_not_treat_box_count_as_accuracy(self):
        decision = decide(metrics(count=100), metrics(count=1), SelectionPolicy(mode="automatic"))
        self.assertEqual(decision["action"], "reject")
        self.assertFalse(decision["accuracy_measured"])

    def test_nan_proxy_and_gold_domain_rejected(self):
        for value in (dict(metrics(), agreement=float("nan")), dict(metrics(), domain="target_gold")):
            with self.assertRaises(ValueError):
                decide(value, None, SelectionPolicy())

    def template(self):
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(dict(CLASS_NAMES=["car", "truck"], SELF_TRAIN={"test": True}, DATA_CONFIG_TAR={},
           DATA_CONFIG={}, MODEL=dict(NAME="SECONDNetIoU", DENSE_HEAD=dict(ANCHOR_GENERATOR_CONFIG=[dict(class_name=c) for c in self.schema.classes])),
           OPTIMIZATION=dict(NUM_EPOCHS=2))))
        return path

    def test_source_config_removes_target_and_adds_ros(self):
        result = source_config(self.template(), self.schema, self.dataset, 13)
        self.assertNotIn("SELF_TRAIN", result)
        self.assertNotIn("DATA_CONFIG_TAR", result)
        self.assertEqual(result["OPTIMIZATION"]["NUM_EPOCHS"], 13)
        self.assertEqual(result["DATA_CONFIG"]["DATA_AUGMENTOR"]["AUG_CONFIG_LIST"][0]["NAME"], "random_object_scaling")

    def test_missing_anchor_for_new_class_blocks_source_train(self):
        unknown = ClassSchema("3d", "cuboid", ("forklift",))
        with self.assertRaisesRegex(ValueError, "anchor"):
            source_config(self.template(), unknown, self.dataset, 1)

    def test_cyclic_config_inheritance_rejected(self):
        path = self.root / "config.yaml"
        path.write_text('_BASE_CONFIG_: config.yaml\n')
        with self.assertRaisesRegex(ValueError, "vòng lặp"):
            load_config(path)

    def test_trend_rule_accepts_non_decreasing_agreement_without_absolute_floor(self):
        policy = SelectionPolicy(mode="automatic")
        v2 = "accepted_pairs_over_confident_boxes_v2"
        first = dict(metrics(count=100, agreement=.31), agreement_definition=v2)
        self.assertEqual(decide(first, None, policy)["action"], "accept")  # baseline, guards only
        legacy = metrics(count=100, agreement=.05)
        self.assertEqual(decide(first, legacy, policy)["action"], "accept")  # other definition = baseline
        self.assertFalse(decide(first, legacy, policy)["compared_with_previous"])
        better = dict(first, agreement=.34)
        self.assertEqual(decide(better, first, policy)["action"], "accept")
        self.assertEqual(decide(dict(first), first, policy)["action"], "accept")  # equal is not a decrease
        worse = dict(first, agreement=.30)
        self.assertEqual(decide(worse, first, policy)["reasons"], ["agreement_decreased"])
        burst = dict(better, box_count=200)
        self.assertIn("box_growth_exceeded", decide(burst, first, policy)["reasons"])
        empty = dict(better, supported_fraction=.5)
        self.assertIn("supported_fraction_below_threshold", decide(empty, first, policy)["reasons"])
        self.assertEqual(decide(better, first, SelectionPolicy())["action"], "review")  # manual mode

    def test_agreement_regression_only_compares_same_definition(self):
        policy = SelectionPolicy(mode="automatic", min_agreement=.3, agreement_rule="threshold")
        old = metrics(count=10, agreement=.9)  # legacy definition: no agreement_definition key
        new = dict(metrics(count=10, agreement=.35), agreement_definition="accepted_pairs_over_confident_boxes_v2")
        self.assertNotIn("agreement_regressed", decide(new, old, policy)["reasons"])
        later = dict(new, agreement=.2)
        self.assertIn("agreement_regressed", decide(later, new, policy)["reasons"])

    def test_fusion_does_not_match_different_classes(self):
        positive, ignored = compare([box()], [box("truck", "b")])
        self.assertFalse(positive)
        self.assertEqual(len(ignored), 2)

    def test_confident_tradeoff_keeps_higher_quality_real_box(self):
        a, b = box(), box(identity="b", x=4.2)
        b.update(confidence=.9, quality=.6)
        positive = compare([a], [b])[0]
        self.assertEqual(positive[0]["id"], "a")
        self.assertEqual(positive[0]["box"], a["box"])
        self.assertEqual(positive[0]["partner_confidence"], .8)
        b["quality"] = .9
        self.assertEqual(compare([a], [b])[0][0]["id"], "b")

    def test_unconfident_partner_is_conflict_even_if_dominated(self):
        a, b = box(), box(identity="b")
        b.update(confidence=.2, quality=.3)
        positive, ignored = compare([a], [b])
        self.assertFalse(positive)
        self.assertEqual({x["reason"] for x in ignored}, {"quality_conflict"})
        a.update(confidence=.5)
        self.assertEqual({x["reason"] for x in compare([a], [b])[1]}, {"low_calibrated_quality"})

    def test_calibration_requires_source_and_correct_checkpoint(self):
        records = dict(domain="source_validation", boxes=[dict(class_name="car", cls_score=i/20, iou_score=i/20, source_gt_iou=i/20) for i in range(12)])
        calibrated = calibrate_source(records, "hash", ["car"])
        value = dict(boxes=[dict(box(), cls_score=.8, iou_score=.8)])
        self.assertIn("quality", normalize(value, calibrated, "hash", ["car"])[0])
        with self.assertRaises(ValueError):
            normalize(value, calibrated, "other", ["car"])
        records["domain"] = "target_gold"
        with self.assertRaises(ValueError):
            calibrate_source(records, "hash", ["car"])

    def test_rare_class_calibration_reports_predictions_without_faking_a_curve(self):
        records=dict(domain='source_validation',boxes=[dict(class_name='motorcycle',
            cls_score=.2,iou_score=.3,source_gt_iou=0.) for _ in range(5)])
        report=source_calibration_report(records,['car','motorcycle'])
        self.assertFalse(report['ready'])
        self.assertEqual([row['samples'] for row in report['classes']],[0,5])
        self.assertEqual(report['classes'][1]['matched_at_iou_0_3'],0)
        with self.assertRaisesRegex(SourceCalibrationError,'motorcycle: 5/10') as error:
            calibrate_source(records,'hash',['car','motorcycle'])
        self.assertEqual(error.exception.report,report)
        records['domain']='target_gold'
        with self.assertRaisesRegex(ValueError,'source validation'):
            source_calibration_report(records,['motorcycle'])

    def test_calibration_does_not_treat_enough_wrong_boxes_as_high_confidence(self):
        records=dict(domain='source_validation',boxes=[dict(class_name='motorcycle',
            cls_score=i/20,iou_score=i/20,source_gt_iou=0.) for i in range(10)])
        calibrated=calibrate_source(records,'hash',['motorcycle'])
        self.assertEqual(calibrated['curves']['motorcycle']['confidence']['y'],[0.]*10)
        self.assertEqual(calibrated['curves']['motorcycle']['quality']['y'],[0.]*10)

    def test_tracking_never_links_two_different_classes(self):
        frames = [dict(frame_id=str(i), sequence_id="s", timestamp=i*.1,
                       boxes=[box("car", "c"), box("truck", "t")]) for i in range(2)]
        result = refine(frames, lambda frame: np.array([[4., 0., 0.]] * 20), self.schema.classes,
                        mode="fixed_sensor", fixed_sensor_confirmed=True)
        tracks = result["frames"][0]["boxes"]
        self.assertNotEqual(tracks[0]["track_id"], tracks[1]["track_id"])
        self.assertEqual({t["class_name"] for t in result["tracks"]}, {"car", "truck"})

    def test_tracking_requires_fixed_sensor_confirmation(self):
        with self.assertRaisesRegex(ValueError, "xác nhận"):
            refine([], lambda frame: None, self.schema.classes, mode="fixed_sensor")

    def test_export_track_only_converts_singletons_and_marks_gap_outside(self):
        frames = [dict(frame_id=str(i), boxes=[dict(box(identity=str(i)), track_id="t")] if i != 1 else []) for i in range(3)]
        document = package(frames, self.schema, "track")
        self.assertFalse(document["shapes"])
        self.assertTrue(document["tracks"][0]["shapes"][1]["outside"])
        self.assertFalse(document["tracks"][0]["shapes"][2]["outside"])

    def test_multiclass_cvat_payload_and_readback(self):
        labels = label_specs(self.schema)
        for index, label in enumerate(labels):
            label["id"] = index + 1
            for aid, attribute in enumerate(label["attributes"]):
                attribute["id"] = index * 10 + aid
        document = package([dict(frame_id="f", boxes=[box(), box("truck", "b")])], self.schema)
        data = payload(document, labels)
        self.assertEqual([s["label_id"] for s in data["shapes"]], [1, 2])
        verify_readback(deepcopy(data), data)
        changed = deepcopy(data)
        changed["shapes"][0]["points"][0] += 1
        with self.assertRaisesRegex(ValueError, "geometry"):
            verify_readback(changed, data)

    def test_binary_pcd_keeps_xyz_without_fake_intensity(self):
        cloud = np.array([[1., 2., 3., 7.]], np.float32)
        header, binary = binary_pcd(cloud).split(b"DATA binary\n")
        self.assertIn(b"FIELDS x y z", header)
        np.testing.assert_array_equal(np.frombuffer(binary, dtype="<f4"), cloud[0, :3])

    def test_two_dimensional_boxes_supported_by_export_contract(self):
        schema = ClassSchema("2d", "bbox", ("person",))
        value = dict(id="a", class_name="person", box=[1., 2., 5., 8.], score=.7)
        document = package([dict(frame_id="f", boxes=[value])], schema)
        self.assertEqual(document["shapes"][0]["type"], "rectangle")


if __name__ == "__main__":
    unittest.main()
