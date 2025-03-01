#!/usr/bin/env python
"""
ChromaDB Document Deletion Tool
This standalone script safely deletes documents from ChromaDB without disrupting the Streamlit app.
"""

import os
import sys
import time
import json
import logging
import traceback
import chromadb
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deletion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("doc_deleter")

def process_deletion_queue(queue_file="delete_queue.txt", chroma_path="./chroma_db", collection_name="rag_vectors"):
    """Process the deletion queue safely without crashing."""
    
    logger.info(f"Starting deletion process at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Atomic queue file handling
        if not os.path.exists(queue_file):
            logger.info("No deletion queue found")
            return True
            
        # Create temp working copy
        temp_queue = f"{queue_file}.working"
        os.rename(queue_file, temp_queue)
        
        with open(temp_queue, "r") as f:
            files_to_delete = [line.strip() for line in f.readlines() if line.strip()]
        
        if not files_to_delete:
            os.remove(temp_queue)
            return True
            
        logger.info(f"Found {len(files_to_delete)} files to delete: {files_to_delete}")
            
        # Create status tracking file
        status = {
            "total": len(files_to_delete),
            "processed": 0,
            "succeeded": [],
            "failed": []
        }
        
        with open("deletion_status.json", "w") as f:
            json.dump(status, f, indent=2)
            
        # Initialize ChromaDB client
        try:
            logger.info(f"Connecting to ChromaDB at {chroma_path}")
            chroma_client = chromadb.PersistentClient(path=chroma_path)
            collection = chroma_client.get_or_create_collection(collection_name)
            logger.info("Successfully connected to ChromaDB")
            
            # Process each file in the queue
            for idx, filename in enumerate(files_to_delete):
                logger.info(f"Processing file {idx+1}/{len(files_to_delete)}: {filename}")
                status["processed"] = idx + 1
                
                try:
                    # Get document IDs for this file
                    all_docs = collection.get(include=['metadatas'])
                    doc_ids = [
                        meta.get("document_id") 
                        for meta in all_docs.get("metadatas", []) 
                        if meta.get("filename") == filename and meta.get("document_id")
                    ]
                    
                    logger.info(f"Found {len(doc_ids)} embeddings for {filename}")
                    
                    # If no doc IDs found, still mark as processed
                    if not doc_ids:
                        logger.info(f"No embeddings found for {filename}, marking as processed")
                        status["succeeded"].append(filename)
                        with open("deletion_status.json", "w") as f:
                            json.dump(status, f, indent=2)
                        continue
                        
                    # Delete each document ID individually
                    success_count = 0
                    for doc_id in doc_ids:
                        try:
                            collection.delete(where={"document_id": doc_id})
                            success_count += 1
                            logger.info(f"Successfully deleted embedding {doc_id}")
                        except Exception as e:
                            logger.error(f"Error deleting embedding {doc_id}: {str(e)}")
                            logger.error(traceback.format_exc())
                    
                    # Log results
                    logger.info(f"Deleted {success_count}/{len(doc_ids)} embeddings for {filename}")
                    
                    # Even if not all were deleted, mark the file as processed
                    status["succeeded"].append(filename)
                    
                except Exception as e:
                    logger.error(f"Failed to process {filename}: {str(e)}")
                    logger.error(traceback.format_exc())
                    status["failed"].append(filename)
                
                # Update status file after each file
                with open("deletion_status.json", "w") as f:
                    json.dump(status, f, indent=2)
                    
            # Update the deletion queue file with only failed deletions
            if status["failed"]:
                with open(queue_file, "w") as f:
                    for filename in status["failed"]:
                        f.write(f"{filename}\n")
                logger.info(f"Updated deletion queue with {len(status['failed'])} failed files")
            else:
                os.remove(temp_queue)
                logger.info("All files processed successfully, removed deletion queue")
                
            logger.info(f"Deletion process completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Successfully deleted {len(status['succeeded'])} files, {len(status['failed'])} files failed")
                
        except Exception as e:
            logger.error(f"Error connecting to ChromaDB: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    except Exception as e:
        logger.error(f"Error processing deletion queue: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    
    return True
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ChromaDB deletion queue")
    parser.add_argument("--queue-file", default="delete_queue.txt", help="Path to deletion queue file")
    parser.add_argument("--chroma-path", default="./chroma_db", help="Path to ChromaDB directory")
    parser.add_argument("--collection", default="rag_vectors", help="Name of ChromaDB collection")
    
    args = parser.parse_args()
    
    process_deletion_queue(
        queue_file=args.queue_file,
        chroma_path=args.chroma_path,
        collection_name=args.collection
    )
