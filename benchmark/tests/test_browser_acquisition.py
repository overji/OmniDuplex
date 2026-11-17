"""Copied browser requests are parsed as data; signed requests are never replayed in bulk."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from benchmark.external.browser_acquisition import read_browser_request
from benchmark.external.acquisition import acquire_one


class BrowserAcquisitionTests(unittest.TestCase):
    def test_only_session_seed_is_extracted_without_executing_shell(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);request=root/'request.txt';marker=root/'must-not-exist'
            request.write_text("curl 'https://www-hj.douyin.com/aweme/v1/web/aweme/detail/?aweme_id=123&a_bogus=fixture' -H 'user-agent: fixture-browser' -b 'sessionid=fixture; test=$(touch "+str(marker)+")'")
            before=request.read_bytes()
            with patch('subprocess.run',side_effect=AssertionError('No shell execution')):
                agent,cookies=read_browser_request(request)
            self.assertEqual(agent,'fixture-browser')
            self.assertEqual(cookies[0]['domain'],'.douyin.com')
            self.assertFalse(marker.exists())
            self.assertEqual(request.read_bytes(),before)

    def test_unrelated_endpoint_and_post_are_rejected(self):
        with TemporaryDirectory() as directory:
            request=Path(directory)/'request.txt'
            for text in ["curl 'https://evil.example/aweme/v1/web/aweme/detail/' -H 'user-agent: fixture' -b 'a=b'",
                         "curl 'https://www.douyin.com/aweme/v1/web/aweme/detail/' -X POST -H 'user-agent: fixture' -b 'a=b'"]:
                request.write_text(text)
                with self.assertRaises(ValueError):read_browser_request(request)

    def test_full_source_is_preserved_without_transcoding(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            def provider(row,work,settings,cookies):
                path=work/'source.mp4';path.write_bytes(b'original-media');return path
            settings=dict(work_dir=str(root/'work'),copy_full_source=True,full_source_copy_tolerance_seconds=.02,duration_tolerance_seconds=.3)
            row=dict(video='videos/test.mp4',source_start_ms=0,source_end_ms=2000)
            with patch('benchmark.external.acquisition.probe_video',return_value=dict(duration=2.,has_audio=True)),patch('subprocess.run',side_effect=AssertionError('No transcode')):
                result=acquire_one(row,root,settings,fetcher=provider)
            self.assertEqual(result['encoding']['mode'],'copy_full_source')
            self.assertEqual((root/'videos/test.mp4').read_bytes(),b'original-media')
            self.assertFalse(result['sha256_checked'])

if __name__=='__main__':unittest.main()
