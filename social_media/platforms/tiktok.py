# social_media/platforms/tiktok.py
import asyncio
import logging
from typing import Dict, List, Optional
import aiohttp
from pathlib import Path

class TikTokClient:
    BASE_URL = "https://open.tiktokapis.com/v2"
    
    def __init__(self, credentials: 'TikTokCredentials'):
        self.credentials = credentials
        self.logger = logging.getLogger(__name__)
        
    async def refresh_access_token(self):
        """Refresh TikTok access token"""
        async with aiohttp.ClientSession() as session:
            data = {
                'client_key': self.credentials.client_key,
                'client_secret': self.credentials.client_secret,
                'grant_type': 'refresh_token',
                'refresh_token': self.credentials.refresh_token
            }
            
            async with session.post(f"{self.BASE_URL}/oauth/token/", json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    self.credentials.access_token = result['access_token']
                    self.credentials.refresh_token = result['refresh_token']
                    return result
                else:
                    raise Exception(f"Token refresh failed: {await response.text()}")

    async def upload_video(self, 
                          video_path: str, 
                          caption: str, 
                          tags: List[str] = None) -> Dict:
        """Upload video to TikTok"""
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # 1. Initialize upload
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.credentials.access_token}',
                    'Content-Type': 'application/json'
                }
                
                # Initialize upload
                init_data = {
                    'post_info': {
                        'title': caption,
                        'privacy_level': 'PUBLIC',
                        'disable_duet': False,
                        'disable_comment': False,
                        'disable_stitch': False
                    }
                }
                
                async with session.post(
                    f"{self.BASE_URL}/video/init/",
                    headers=headers,
                    json=init_data
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Upload initialization failed: {await response.text()}")
                    
                    init_result = await response.json()
                    upload_url = init_result['data']['upload_url']

                # 2. Upload video file
                with open(video_path, 'rb') as video_file:
                    async with session.put(
                        upload_url,
                        data=video_file,
                        headers={'Content-Type': 'video/mp4'}
                    ) as upload_response:
                        if upload_response.status != 200:
                            raise Exception(f"Video upload failed: {await upload_response.text()}")

                # 3. Publish video
                publish_data = {
                    'upload_id': init_result['data']['upload_id'],
                    'post_info': {
                        'title': caption,
                        'tags': tags or []
                    }
                }
                
                async with session.post(
                    f"{self.BASE_URL}/video/publish/",
                    headers=headers,
                    json=publish_data
                ) as publish_response:
                    if publish_response.status != 200:
                        raise Exception(f"Video publishing failed: {await publish_response.text()}")
                    
                    result = await publish_response.json()
                    return {
                        "status": "success",
                        "platform": "tiktok",
                        "video_path": str(video_path),
                        "caption": caption,
                        "tags": tags,
                        "video_id": result['data']['video_id']
                    }

        except Exception as e:
            self.logger.error(f"Failed to upload to TikTok: {str(e)}")
            if 'token_expired' in str(e).lower():
                try:
                    await self.refresh_access_token()
                    # Retry upload once after token refresh
                    return await self.upload_video(video_path, caption, tags)
                except Exception as refresh_error:
                    raise Exception(f"Token refresh failed: {str(refresh_error)}")
            raise