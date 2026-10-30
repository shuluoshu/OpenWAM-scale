"""Serve the existing report root with byte ranges for seekable MP4 playback."""
import argparse
from pathlib import Path, PurePosixPath
from aiohttp import web


def application(directory):
    root=Path(directory).absolute()
    async def handle(request):
        relative=request.match_info['path']
        if relative.startswith('/') or '..' in PurePosixPath(relative).parts:
            raise web.HTTPNotFound()
        # Existing report routes are intentional symlinks into report folders.
        path=root/relative
        if path.is_dir():path=path/'index.html'
        if not path.is_file():raise web.HTTPNotFound()
        return web.FileResponse(path,headers={'Cache-Control':'no-cache'})
    app=web.Application()
    app.router.add_get('/{path:.*}',handle)
    return app


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True,type=Path)
    p.add_argument('--port',type=int,default=18082);a=p.parse_args()
    web.run_app(application(a.directory),host='0.0.0.0',port=a.port)
