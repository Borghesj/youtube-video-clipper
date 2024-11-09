# upload_queue/queue_manager.py
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

class UploadQueue:
    MAX_RETRIES = 3
    RETRY_DELAY = 300  # 5 minutes
    
    def __init__(self, storage_dir: str = "upload_queue/storage"):
        self.storage_dir = Path(storage_dir)
        self.pending_dir = self.storage_dir / "pending"
        self.completed_dir = self.storage_dir / "completed"
        self.failed_dir = self.storage_dir / "failed"
        self.processing_dir = self.storage_dir / "processing"
        
        # Create directories
        for directory in [self.pending_dir, self.completed_dir, 
                         self.failed_dir, self.processing_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def add_to_queue(self, 
                     video_path: str, 
                     platforms: List[str],
                     caption: str,
                     schedule_time: Optional[datetime] = None,
                     tags: List[str] = None) -> str:
        """Add a video to the upload queue"""
        queue_item = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "video_path": video_path,
            "platforms": platforms,
            "caption": caption,
            "tags": tags or [],
            "schedule_time": schedule_time.isoformat() if schedule_time else None,
            "status": "pending",
            "retry_count": 0,
            "next_retry": None,
            "created_at": datetime.now().isoformat(),
            "modified_at": datetime.now().isoformat()
        }
        
        queue_file = self.pending_dir / f"{queue_item['id']}.json"
        with open(queue_file, 'w') as f:
            json.dump(queue_item, f, indent=2)
            
        return queue_item['id']
    
    def move_to_processing(self, item_id: str) -> Dict:
        """Move item to processing state"""
        src_file = self.pending_dir / f"{item_id}.json"
        dst_file = self.processing_dir / f"{item_id}.json"
        
        with open(src_file) as f:
            item = json.load(f)
            
        item['status'] = 'processing'
        item['modified_at'] = datetime.now().isoformat()
        
        with open(dst_file, 'w') as f:
            json.dump(item, f, indent=2)
            
        os.remove(src_file)
        return item
    
    def mark_completed(self, item_id: str, result: Dict):
        """Mark item as completed"""
        src_file = self.processing_dir / f"{item_id}.json"
        dst_file = self.completed_dir / f"{item_id}.json"
        
        with open(src_file) as f:
            item = json.load(f)
            
        item['status'] = 'completed'
        item['result'] = result
        item['completed_at'] = datetime.now().isoformat()
        item['modified_at'] = datetime.now().isoformat()
        
        with open(dst_file, 'w') as f:
            json.dump(item, f, indent=2)
            
        os.remove(src_file)
        
    def mark_failed(self, item_id: str, error: str):
        """Mark item as failed or schedule retry"""
        src_file = self.processing_dir / f"{item_id}.json"
        
        with open(src_file) as f:
            item = json.load(f)
            
        item['retry_count'] = item.get('retry_count', 0) + 1
        item['last_error'] = error
        item['modified_at'] = datetime.now().isoformat()
        
        if item['retry_count'] < self.MAX_RETRIES:
            # Schedule retry
            retry_time = datetime.now() + timedelta(seconds=self.RETRY_DELAY)
            item['status'] = 'pending'
            item['next_retry'] = retry_time.isoformat()
            dst_file = self.pending_dir / f"{item_id}.json"
        else:
            # Mark as failed
            item['status'] = 'failed'
            dst_file = self.failed_dir / f"{item_id}.json"
            
        with open(dst_file, 'w') as f:
            json.dump(item, f, indent=2)
            
        os.remove(src_file)