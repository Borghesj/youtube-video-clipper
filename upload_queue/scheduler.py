# upload_queue/scheduler.py
import asyncio
import logging
from datetime import datetime
from typing import Dict

from social_media.platforms.tiktok import TikTokClient
from upload_queue.queue_manager import UploadQueue

class UploadScheduler:
    def __init__(self):
        self.queue = UploadQueue()
        self.logger = logging.getLogger(__name__)
        self.tiktok_client = TikTokClient()
    
    async def process_upload(self, item: Dict):
        """Process a single upload"""
        try:
            if 'tiktok' in item['platforms']:
                await self.tiktok_client.upload_video(
                    video_path=item['video_path'],
                    caption=item['caption']
                )
                
            # Move to completed
            # Implementation here
            
        except Exception as e:
            self.logger.error(f"Upload failed: {str(e)}")
            # Move to failed
            # Implementation here
    
    async def run(self):
        """Main scheduler loop"""
        while True:
            try:
                pending = self.queue.get_pending_uploads()
                current_time = datetime.now()
                
                for item in pending:
                    schedule_time = datetime.fromisoformat(item['schedule_time']) if item['schedule_time'] else None
                    
                    if not schedule_time or schedule_time <= current_time:
                        await self.process_upload(item)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Scheduler error: {str(e)}")
                await asyncio.sleep(60)