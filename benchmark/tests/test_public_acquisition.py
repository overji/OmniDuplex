"""Acquisition does not hash or overwrite videos; explicit provenance stays intact."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.external.acquisition import acquire_one, actual_platform, canonical_url, failure_reason, private_cookie_file


class PublicAcquisition(unittest.TestCase):
    def test_cookie_header_is_converted_privately_without_changing_source(self):
        from http.cookiejar import MozillaCookieJar
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input.txt';source.write_text('Cookie: ttwid=fixture; s_v_web_id=fixture2')
            before=source.read_bytes();work=root/'work';work.mkdir()
            private=private_cookie_file(source,work)
            jar=MozillaCookieJar(str(private));jar.load(ignore_discard=True)
            self.assertEqual(len(jar),2)
            self.assertEqual(private.stat().st_mode&0o777,0o600)
            self.assertEqual(source.read_bytes(),before)

    def test_repair_duplicated_bilibili_url_without_mutating_record(self):
        url='https://www.bilibili.com/video/https://www.bilibili.com/video/BV1Tz7X6uEP1//'
        row=dict(source='bilibili',original_video_url=url)
        self.assertEqual(canonical_url(row),'https://www.bilibili.com/video/BV1Tz7X6uEP1/')
        self.assertEqual(row['original_video_url'],url)

    def test_url_host_overrides_wrong_source_label(self):
        row=dict(source='xiaohongshu',original_video_url='https://v.douyin.com/abc/')
        self.assertEqual(actual_platform(row),'douyin')

    def test_existing_media_is_never_overwritten_or_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'videos').mkdir();target=root/'videos/v.mp4';target.write_bytes(b'original')
            with patch('benchmark.external.acquisition.fetch_source',side_effect=AssertionError):
                result=acquire_one(dict(video='videos/v.mp4'),root,{})
            self.assertEqual(result['status'],'existing')
            self.assertFalse(result['sha256_checked'])
            self.assertEqual(target.read_bytes(),b'original')

    def test_cookie_requirement_has_nonsecret_reason(self):
        self.assertEqual(failure_reason(RuntimeError('Fresh cookies needed: secret-cookie-value')),'cookies_required')

if __name__=='__main__':unittest.main()
