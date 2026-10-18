"""Backend CLI for inspection, projects, pretrain plans and monitored jobs."""
import argparse
import json
from pathlib import Path
import yaml
from . import jobs
from .device import probe
from .inventory import inspect_dataset
from .io import read
from .pretrain import source_config
from .projects import ProjectStore
from .schema import discover
from .selection import SelectionPolicy
from .settings import Settings


def main(argv=None):
    parser = argparse.ArgumentParser(prog="annoloop")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("doctor")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("dataset")
    create = sub.add_parser("create")
    create.add_argument("dataset")
    create.add_argument("--name", required=True)
    create.add_argument("--rounds", type=int, default=3)
    create.add_argument("--execution", choices=["manual", "automatic"], default="manual")
    create.add_argument("--selection", choices=["manual", "automatic"], default="manual")
    sub.add_parser("projects")
    show = sub.add_parser("show")
    show.add_argument("project")
    select = sub.add_parser("select")
    select.add_argument("project")
    select.add_argument("version")
    finish = sub.add_parser("finish")
    finish.add_argument("project")
    plan = sub.add_parser("source-config")
    plan.add_argument("dataset")
    plan.add_argument("--template", required=True)
    plan.add_argument("--source", required=True)
    plan.add_argument("--epochs", required=True, type=int)
    plan.add_argument("--ros", nargs=2, type=float, default=[.9, 1.1])
    plan.add_argument("--no-ros", action="store_true")
    plan.add_argument("--output", required=True)
    launch = sub.add_parser("pretrain")
    launch.add_argument("spec")
    pair = sub.add_parser("pretrain-pair")
    pair.add_argument("spec")
    status = sub.add_parser("job")
    status.add_argument("path")
    cancel = sub.add_parser("stop-job")
    cancel.add_argument("path")
    options = parser.parse_args(argv)
    settings = Settings.load()
    store = ProjectStore(settings.data)
    if options.action == "doctor":
        result = dict(python=str(settings.python), cuda=probe(settings.python),
                      training_launcher=str(settings.training_launcher), data=str(settings.data))
    elif options.action == "inspect":
        result = inspect_dataset(options.dataset)
    elif options.action == "create":
        result = dict(project=store.create(options.name, options.dataset, options.rounds, options.execution,
                                          SelectionPolicy(mode=options.selection)))
    elif options.action == "projects":
        result = store.list()
    elif options.action == "show":
        result = store.get(options.project)
    elif options.action == "select":
        store.select(options.project, options.version)
        result = store.get(options.project)
    elif options.action == "finish":
        store.request_finish(options.project)
        result = store.get(options.project)
    elif options.action == "source-config":
        schema, _ = discover(options.dataset)
        result = source_config(options.template, schema, options.source, options.epochs,
                               None if options.no_ros else options.ros)
        path = Path(options.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            yaml.safe_dump(result, stream, sort_keys=False)
    elif options.action == "pretrain":
        result = dict(job=jobs.start_pretrain(read(options.spec), settings))
    elif options.action == "pretrain-pair":
        result = dict(pair=jobs.start_source_pair(read(options.spec), settings))
    elif options.action == "job":
        result = jobs.status(options.path)
    else:
        result = dict(stop_sent=jobs.stop(options.path))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
