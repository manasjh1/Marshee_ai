#!/usr/bin/env python3
"""
Simple Embeddings Creator for Marshee Dog Health System

Usage:
    python create_embeddings_simple.py

Just modify the CONFIG section below to specify:
- Which folder to process
- Which namespace to use
- Whether to clear existing embeddings

No endpoints, no complexity - just run and it works!
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# ============================================
# 🎯 CONFIGURATION - MODIFY THESE SETTINGS
# ============================================

# Which folder to process (relative to script location)
FOLDER_TO_PROCESS = "data\\all_files"  # Change this to your folder
# Which namespace to use in Pinecone
TARGET_NAMESPACE = "dog-health-knowledge"  # Change this if needed

CLEAR_EXISTING = False  

DOCUMENT_TYPE = "knowledge"

def create_embeddings():
    """Main function to create embeddings"""
    
    print("🧠 Simple Embeddings Creator for Marshee")
    print("=" * 50)
    print(f"📁 Processing folder: {FOLDER_TO_PROCESS}")
    print(f"🎯 Target namespace: {TARGET_NAMESPACE}")
    print(f"🧹 Clear existing: {CLEAR_EXISTING}")
    print(f"📝 Document type: {DOCUMENT_TYPE}")
    print("=" * 50)
    
    try:
        # Import required services
        from services.embedding_service import GeminiEmbeddingService
        from services.vector_db_service import PineconeVectorDB
        from modals.document import DocumentChunk, DocumentType
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        print("✅ Services imported successfully")
        
        # Initialize services
        print("🔄 Initializing services...")
        embedding_service = GeminiEmbeddingService()
        vector_db = PineconeVectorDB()
        
        # Text splitter for chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        print("✅ Services initialized")
        
        # Check if namespace exists and create if needed
        print(f"🔍 Checking if namespace '{TARGET_NAMESPACE}' exists...")
        try:
            # Try to check if namespace exists (this will depend on your vector_db implementation)
            # Most vector databases don't require explicit namespace creation, but we'll try to ensure it exists
            
            # First, try a simple operation to see if namespace is accessible
            test_embedding = [0.0] * 1536  # Typical embedding dimension, adjust if needed
            
            # Try to query the namespace - if it doesn't exist, this might fail or return empty
            test_results = vector_db.similarity_search(
                query_embedding=test_embedding,
                top_k=1,
                namespace=TARGET_NAMESPACE
            )
            
            print(f"✅ Namespace '{TARGET_NAMESPACE}' is accessible")
            
        except Exception as e:
            print(f"⚠️ Namespace check failed: {e}")
            print(f"🔧 Attempting to create/initialize namespace '{TARGET_NAMESPACE}'...")
            
            # For Pinecone, namespaces are created implicitly when you first upsert data
            # But we can try to create an empty upsert to initialize it
            try:
                # Create a dummy embedding to initialize the namespace
                dummy_chunk = DocumentChunk(
                    content="initialization",
                    metadata={
                        "initialization": True,
                        "namespace": TARGET_NAMESPACE,
                        "created_at": str(int(time.time()))
                    },
                    document_type=DocumentType.KNOWLEDGE,
                    namespace=TARGET_NAMESPACE,
                    original_namespace=TARGET_NAMESPACE
                )
                
                # Create embedding for dummy content
                dummy_embedded = embedding_service.embed_document_chunks([dummy_chunk])
                
                if dummy_embedded:
                    # Upsert the dummy data to create namespace
                    vector_db.upsert_chunks(dummy_embedded, namespace=TARGET_NAMESPACE)
                    print(f"✅ Namespace '{TARGET_NAMESPACE}' initialized successfully")
                    
                    # Immediately delete the dummy data
                    time.sleep(1)  # Brief pause to ensure upsert completes
                    # Note: You might want to implement a method to delete specific vectors by ID
                    # For now, we'll leave this as the namespace is initialized
                
            except Exception as init_error:
                print(f"⚠️ Could not initialize namespace: {init_error}")
                print("💡 Proceeding anyway - namespace will be created when first data is uploaded")
        
        # Check if folder exists
        folder_path = Path(FOLDER_TO_PROCESS)
        if not folder_path.exists():
            print(f"❌ Error: Folder '{FOLDER_TO_PROCESS}' does not exist!")
            print("💡 Please create the folder and add your .txt files")
            return
        
        # Get all text files
        txt_files = list(folder_path.glob("*.txt"))
        if not txt_files:
            print(f"❌ Error: No .txt files found in '{FOLDER_TO_PROCESS}'!")
            print("💡 Please add some .txt files to process")
            return
        
        print(f"📄 Found {len(txt_files)} .txt files to process")
        
        # Clear existing embeddings if requested
        if CLEAR_EXISTING:
            print(f"🧹 Clearing existing embeddings in namespace '{TARGET_NAMESPACE}'...")
            success = vector_db.delete_namespace(TARGET_NAMESPACE)
            if success:
                print("✅ Existing embeddings cleared")
                # After clearing, we'll need to ensure namespace exists again
                print(f"🔧 Re-initializing namespace after clearing...")
            else:
                print("⚠️ Warning: Could not clear existing embeddings")
            time.sleep(2)  # Wait for deletion to complete
        
        # Process each file
        total_chunks = 0
        processed_files = 0
        
        for i, txt_file in enumerate(txt_files, 1):
            print(f"\n📄 Processing file {i}/{len(txt_files)}: {txt_file.name}")
            
            try:
                # Read file content
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content.strip():
                    print(f"⚠️ Skipping empty file: {txt_file.name}")
                    continue
                
                # Split into chunks
                text_chunks = text_splitter.split_text(content)
                print(f"   📝 Split into {len(text_chunks)} chunks")
                
                # Create DocumentChunk objects
                document_chunks = []
                for chunk_idx, chunk_content in enumerate(text_chunks):
                    if chunk_content.strip():
                        metadata = {
                            "filename": txt_file.name,
                            "chunk_index": chunk_idx,
                            "total_chunks": len(text_chunks),
                            "document_type": DOCUMENT_TYPE,
                            "namespace": TARGET_NAMESPACE,
                            "source": txt_file.stem,  # filename without extension
                            "file_path": str(txt_file),
                            "created_at": str(int(time.time()))
                        }
                        
                        chunk = DocumentChunk(
                            content=chunk_content.strip(),
                            metadata=metadata,
                            document_type=DocumentType.KNOWLEDGE if DOCUMENT_TYPE == "knowledge" else DocumentType.PRODUCT,
                            namespace=TARGET_NAMESPACE,
                            original_namespace=TARGET_NAMESPACE
                        )
                        document_chunks.append(chunk)
                
                if not document_chunks:
                    print(f"⚠️ No valid chunks created from {txt_file.name}")
                    continue
                
                # Create embeddings
                print(f"   🧠 Creating embeddings for {len(document_chunks)} chunks...")
                embedded_chunks = embedding_service.embed_document_chunks(document_chunks)
                
                if not embedded_chunks:
                    print(f"❌ Failed to create embeddings for {txt_file.name}")
                    continue
                
                print(f"   ✅ Created {len(embedded_chunks)} embeddings")
                
                # Upload to vector database
                print(f"   📤 Uploading to namespace '{TARGET_NAMESPACE}'...")
                try:
                    success = vector_db.upsert_chunks(embedded_chunks, namespace=TARGET_NAMESPACE)
                    
                    if success:
                        total_chunks += len(embedded_chunks)
                        processed_files += 1
                        print(f"   ✅ Successfully uploaded {len(embedded_chunks)} chunks")
                        
                        # If this is the first successful upload, confirm namespace creation
                        if total_chunks == len(embedded_chunks):
                            print(f"   🎉 Namespace '{TARGET_NAMESPACE}' created and populated!")
                    else:
                        print(f"   ❌ Failed to upload chunks for {txt_file.name}")
                        
                except Exception as upload_error:
                    print(f"   ❌ Upload error for {txt_file.name}: {upload_error}")
                    # If it's a namespace-related error, try to create namespace first
                    if "namespace" in str(upload_error).lower():
                        print(f"   🔧 Retrying upload after namespace initialization...")
                        try:
                            success = vector_db.upsert_chunks(embedded_chunks, namespace=TARGET_NAMESPACE)
                            if success:
                                total_chunks += len(embedded_chunks)
                                processed_files += 1
                                print(f"   ✅ Retry successful! Uploaded {len(embedded_chunks)} chunks")
                            else:
                                print(f"   ❌ Retry failed for {txt_file.name}")
                        except Exception as retry_error:
                            print(f"   ❌ Retry error: {retry_error}")
                
            except Exception as e:
                print(f"❌ Error processing {txt_file.name}: {e}")
                continue
        
        # Final results
        print("\n" + "=" * 50)
        print("🎉 PROCESSING COMPLETE!")
        print("=" * 50)
        print(f"✅ Successfully processed: {processed_files}/{len(txt_files)} files")
        print(f"📊 Total chunks created: {total_chunks}")
        print(f"🎯 Stored in namespace: {TARGET_NAMESPACE}")
        
        if processed_files < len(txt_files):
            failed_count = len(txt_files) - processed_files
            print(f"⚠️ Failed to process: {failed_count} files")
        
        # Test search (optional)
        if total_chunks > 0:
            print(f"\n🔍 Testing search functionality...")
            test_query = "dog nutrition"  # You can change this
            query_embedding = embedding_service.create_single_embedding(test_query)
            
            if query_embedding:
                search_results = vector_db.similarity_search(
                    query_embedding=query_embedding,
                    top_k=3,
                    namespace=TARGET_NAMESPACE
                )
                
                if search_results:
                    print(f"✅ Search test successful! Found {len(search_results)} results for '{test_query}'")
                    print(f"   Top result score: {search_results[0].get('score', 0):.3f}")
                else:
                    print("⚠️ Search test returned no results")
            else:
                print("⚠️ Could not create test query embedding")
        
        print(f"\n🚀 Your embeddings are ready! You can now use the chat system.")
        print(f"💬 The RAG system will search the '{TARGET_NAMESPACE}' namespace for relevant information.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you have installed all requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Check your .env file and make sure all API keys are set correctly")

def show_config_help():
    """Show help for configuration"""
    print("\n📋 CONFIGURATION HELP:")
    print("=" * 30)
    print("🎯 To process different folders, modify these variables at the top of this file:")
    print("")
    print("FOLDER_TO_PROCESS examples:")
    print('  "data/knowledge"     # For health & care knowledge')
    print('  "data/products"      # For product information')
    print('  "data/all_files"     # For mixed content')
    print('  "my_custom_folder"   # For any custom folder')
    print("")
    print("TARGET_NAMESPACE examples:")
    print('  "dog-health-knowledge"  # For health information')
    print('  "marshee-products"      # For product information')
    print('  "my-custom-namespace"   # For custom content')
    print("")
    print("CLEAR_EXISTING:")
    print("  True   # Delete existing embeddings first")
    print("  False  # Keep existing embeddings")
    print("")
    print("DOCUMENT_TYPE:")
    print('  "knowledge"  # For educational content')
    print('  "product"    # For product information')

if __name__ == "__main__":
    print("🐕 Marshee Dog Health System - Simple Embeddings Creator")
    
    # Show help if requested
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_config_help()
        sys.exit(0)
    
    # Check if .env file exists
    if not Path('.env').exists():
        print("❌ Error: .env file not found!")
        print("💡 Please create a .env file with your API keys")
        sys.exit(1)
    
    # Run the embeddings creation
    create_embeddings()