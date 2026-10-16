"""Authentication is independent of final prediction and must never create tasks."""
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch
from annoloop.cvat import check_login


class LoginTests(unittest.TestCase):
    def test_authentication_and_whoami_without_task_operations_or_secret_receipt(self):
        client=MagicMock()
        client.users.retrieve_current_user.return_value=SimpleNamespace(username='fixture_user',id=7)
        context=MagicMock();context.__enter__.return_value=client
        with patch('cvat_sdk.make_client',return_value=context) as connect:
            result=check_login('http://localhost:8080/','fixture_user','fixture_secret')
            connect.assert_called_once_with('http://localhost:8080',credentials=('fixture_user','fixture_secret'))
            client.users.retrieve_current_user.assert_called_once_with()
            client.tasks.create.assert_not_called()
            client.tasks.list.assert_not_called()
        self.assertEqual(result['username'],'fixture_user')
        self.assertEqual(result['user_id'],7)
        self.assertNotIn('fixture_secret',str(result))
        self.assertEqual(set(result),{'host','username','user_id','checked_at'})

    def test_bad_credentials_show_auth_status_without_echoing_response_or_password(self):
        class Rejected(Exception):status=401
        with patch('cvat_sdk.make_client',side_effect=Rejected('response contains fixture_secret')):
            with self.assertRaisesRegex(ValueError,'HTTP 401') as caught:
                check_login('http://localhost:8080','fixture_user','fixture_secret')
            self.assertNotIn('fixture_secret',str(caught.exception))

    def test_empty_fields_and_embedded_credentials_do_not_contact_server(self):
        for host,user,password in [('http://localhost:8080','','secret'),
                                    ('http://localhost:8080','user',''),
                                    ('http://user:secret@localhost:8080','user','secret'),
                                    ('localhost:8080','user','secret')]:
            with patch('cvat_sdk.make_client') as connect:
                with self.assertRaises(ValueError):check_login(host,user,password)
                connect.assert_not_called()

    def test_unreachable_server_is_not_reported_as_wrong_password(self):
        with patch('cvat_sdk.make_client',side_effect=ConnectionError('do not echo fixture_secret')):
            with self.assertRaisesRegex(ValueError,'ConnectionError') as caught:
                check_login('http://localhost:8080','fixture_user','fixture_secret')
            self.assertNotIn('fixture_secret',str(caught.exception))
