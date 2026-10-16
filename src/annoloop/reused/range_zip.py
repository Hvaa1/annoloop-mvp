"""Bounded ZIP-member transport; adapted from demo experiments/fetch_secondiou.py.

No whole-archive fallback, extraction to archive paths or arbitrary source URL.
Final member hash is checked by the caller, independently of ZIP CRC.
"""
import html
import io
import re
import urllib.parse
import urllib.request


def google_url(file_id,opener=urllib.request.urlopen):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',file_id):raise ValueError('Google file ID không hợp lệ')
    url='https://drive.usercontent.google.com/download?'+urllib.parse.urlencode(dict(id=file_id,export='download'))
    with opener(urllib.request.Request(url,headers={'Range':'bytes=0-0'}),timeout=30) as response:
        if response.status==206:return url
        if response.status!=200:raise ValueError('Không mở được archive')
        page=response.read(65537)
        if len(page)>65536:raise ValueError('Server không trả range hoặc form xác nhận nhỏ')
    page=page.decode('utf-8');form=re.search(r'<form[^>]*action="([^"]+)"',page)
    if not form:raise ValueError('Không tìm thấy form xác nhận Google Drive; link có thể hết hạn')
    action=html.unescape(form[1]);parsed=urllib.parse.urlsplit(action)
    if parsed.scheme!='https' or parsed.hostname!='drive.usercontent.google.com' or parsed.username or parsed.port not in (None,443):
        raise ValueError('Host xác nhận không hợp lệ')
    fields=dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"',page))
    return action+'?'+urllib.parse.urlencode({k:html.unescape(v) for k,v in fields.items()})


class RemoteZip(io.RawIOBase):
    def __init__(self,url,opener=urllib.request.urlopen):
        self.url=url;self.opener=opener;self.position=0;self.downloaded_bytes=0;self.etag=None
        with opener(urllib.request.Request(url,headers={'Range':'bytes=0-0'}),timeout=30) as response:
            match=re.fullmatch(r'bytes 0-0/(\d+)',response.headers.get('Content-Range',''))
            if response.status!=206 or not match:raise ValueError('Server không hỗ trợ HTTP range; không tải cả ZIP')
            self.size=int(match[1]);self.etag=response.headers.get('ETag')
            if self.size<22:raise ValueError('Archive quá nhỏ')

    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        pos=offset if whence==0 else self.position+offset if whence==1 else self.size+offset if whence==2 else -1
        if not 0<=pos<=self.size:raise ValueError('ZIP seek ngoài archive')
        self.position=pos;return pos

    def read(self,size=-1):
        count=min(self.size-self.position,size) if size>=0 else self.size-self.position
        if count>16*1024*1024:raise ValueError('ZIP yêu cầu đọc quá lớn; không tải cả archive')
        parts=[]
        while count:
            amount=min(count,4*1024*1024);start=self.position
            headers={'Range':f'bytes={start}-{start+amount-1}'}
            if self.etag and not self.etag.startswith('W/'):headers['If-Match']=self.etag
            with self.opener(urllib.request.Request(self.url,headers=headers),timeout=45) as response:
                if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{start+amount-1}/{self.size}':
                    raise ValueError('HTTP range không khớp')
                if self.etag and response.headers.get('ETag')!=self.etag:raise ValueError('Archive đã đổi trong lúc tải')
                data=response.read(amount+1)
            if len(data)!=amount:raise ValueError('HTTP range thiếu dữ liệu')
            self.position+=amount;self.downloaded_bytes+=amount;count-=amount;parts.append(data)
        return b''.join(parts)
