import streamlit as st
import time
from twelvelabs import TwelveLabs
from PIL import Image
import pandas as pd
from urllib.parse import urlparse
import uuid
from dotenv import load_dotenv
import os
from pymilvus import MilvusClient
from pymilvus import connections
from pymilvus import (
   FieldSchema, DataType, 
   CollectionSchema, Collection,
   utility
)

load_dotenv()

TWELVELABS_API_KEY = os.getenv('TWELVELABS_API_KEY')
MILVUS_DB_NAME = os.getenv('MILVUS_DB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
MILVUS_HOST = os.getenv('MILVUS_HOST')
MILVUS_PORT = os.getenv('MILVUS_PORT')
URL = os.getenv('URL')
TOKEN = os.getenv('TOKEN')

# Connect to Milvus
connections.connect(
   uri=URL,
   token=TOKEN
)

# Define fields for schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
]

# Create schema with dynamic fields for metadata
schema = CollectionSchema(
    fields=fields,
    enable_dynamic_field=True
)

# Check if collection exists
if utility.has_collection(COLLECTION_NAME):
    # If exists, just load the existing collection
    collection = Collection(COLLECTION_NAME)
    print(f"Using existing collection: {COLLECTION_NAME}")
else:
    # If doesn't exist, create new collection
    collection = Collection(COLLECTION_NAME, schema)
    print(f"Created new collection: {COLLECTION_NAME}")
    
    # Create index for new collection
    if not collection.has_index():
        collection.create_index(
            field_name="vector",
            index_params={
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
        )
        print("Created index for the new collection")

# Load collection for searching
collection.load()

# Set the milvus_client to the collection
milvus_client = collection

# st.write(f"Connected to collection: {COLLECTION_NAME}")


# # Initialize Milvus client
# milvus_client = MilvusClient(
#     uri=URL,
#     token=TOKEN
# )

# collection_name = COLLECTION_NAME

# # Check if collection exists and drop if necessary
# if milvus_client.has_collection(collection_name):
#     milvus_client.drop_collection(collection_name)

# # Create collection with proper schema
# milvus_client.create_collection(
#     collection_name=collection_name,
#     dimension=1024,
#     vector_field_name="vector",
#     enable_dynamic_field=True
# )

# # Create index
# milvus_client.create_index(
#     collection_name=collection_name,
#     field_name="vector",
#     index_params={
#         "metric_type": "COSINE",
#         "index_type": "IVF_FLAT",
#         "params": {"nlist": 128}
#     }
# )

# # Load collection
# milvus_client.load_collection(collection_name)

st.write(f"Collection '{COLLECTION_NAME}' created successfully")
st.write("Hello")



# Add these functions after your existing imports and setup code

def generate_embedding(video_url, product_info):
    """Generate embeddings for a video URL with associated product information"""
    try:
        st.write(f"Processing video for product: {product_info['title']}")
        st.write(f"Video URL: {video_url}")

        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)

        task = twelvelabs_client.embed.task.create(
            model_name="Marengo-retrieval-2.7",
            video_url=video_url
        )

        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        def on_task_update(task):
            status_text.write(f"Status: {task.status}")
            if task.status == "completed":
                progress_bar.progress(100)
            elif task.status == "processing":
                progress_bar.progress(50)

        status = task.wait_for_done(
            sleep_interval=2,
            callback=on_task_update
        )

        task_result = twelvelabs_client.embed.task.retrieve(task.id)

        embeddings = []
        for segment in task_result.video_embedding.segments:
            embeddings.append({
                'embedding': segment.embeddings_float,
                'start_offset_sec': segment.start_offset_sec,
                'end_offset_sec': segment.end_offset_sec,
                'embedding_scope': segment.embedding_scope,
                'video_url': video_url,
                'product_id': product_info['product_id'],
                'title': product_info['title'],
                'description': product_info['desc'],
                'link': product_info['link']
            })

        return embeddings, task_result, None
    except Exception as e:
        return None, None, str(e)

def insert_embeddings(collection, embeddings):
    """Insert embeddings into Milvus collection"""
    try:
        data = []
        for i, emb in enumerate(embeddings):
            data.append({
                "id": int(uuid.uuid4().int & (1<<63)-1),  # Generate unique ID
                "vector": emb['embedding'],
                "metadata": {
                    "scope": emb['embedding_scope'],
                    "start_time": emb['start_offset_sec'],
                    "end_time": emb['end_offset_sec'],
                    "video_url": emb['video_url'],
                    "product_id": emb['product_id'],
                    "title": emb['title'],
                    "description": emb['description'],
                    "link": emb['link']
                }
            })

        insert_result = collection.insert(data)
        return insert_result
    except Exception as e:
        st.error(f"Error inserting embeddings: {str(e)}")
        return None

def process_products():
    """Process all products and insert embeddings into Milvus"""
    json_data={
  "products": [
    {
      "product_id": "1996777",
      "title": "Manscaping Guide",
      "desc": "Manscaping is no more a hush-hush thing to talk about, it's a personal choice if you wanna get rid of your body hair or not to help out with this in this today's episode of You Got This Bro here are some tips for you so that can have smooth and fuss-free experience. Watch now! Tap to shop products.",
      "link": "https://www.myntra.com/1996777",
      "video_url": "https://storage.googleapis.com/test002-weavite/mens%20shirt.mp4"
    },
    {
      "product_id": "31014584",
      "title": "Black Women Leather Biker Jacket",
      "desc": "Get edgy with our black women leather biker jacket. Featuring lapel collar, belt, and premium leather, perfect for a chic, mysterious look.Embrace the biker chic with our stunning black women leather biker jacket. Crafted from high-quality leather, this stylish design boasts: Lapel collar for added",
      "link": "https://www.myntra.com/31014584",
      "video_url": "https://storage.googleapis.com/test002-weavite/Black_Women_Leather_Biker_Jacket.mp4"
    },
    {
      "product_id": "19482596",
      "title": "Sakhi - Bridesmaid Collection",
      "desc": "With the wedding season around the corner, we bring to you \"Sakhi\" - the ultimate destination to all your bridesmaid fashion needs.",
      "link": "https://www.myntra.com/19482596",
      "video_url": "https://storage.googleapis.com/test002-weavite/Sakhi_Bridesmaid_Collection.mp4"
    }
  ]
}
    all_embeddings = []
    total_products = len(json_data['products'])

    # Create a container for product processing status
    status_container = st.container()

    for idx, product in enumerate(json_data['products'], 1):
        with status_container:
            st.write(f"\n--- Processing product {idx}/{total_products} ---")
            st.write(f"Title: {product['title']}")
            st.write(f"Product ID: {product['product_id']}")
            
            try:
                # Generate embeddings
                embeddings, task_result, error = generate_embedding(
                    product['video_url'],
                    product
                )

                if error:
                    st.error(f"Error processing video: {error}")
                    continue

                if embeddings:
                    all_embeddings.extend(embeddings)
                    st.success(f"Successfully generated {len(embeddings)} embeddings")

                    # Insert embeddings into Milvus
                    insert_result = insert_embeddings(
                        collection,
                        embeddings
                    )

                    if insert_result:
                        st.success(f"Successfully inserted embeddings into Milvus")
                    else:
                        st.error("Failed to insert embeddings into Milvus")

                    # Display sample embeddings
                    with st.expander("View sample embeddings"):
                        for i, emb in enumerate(embeddings[:2]):
                            st.write(f"\nEmbedding {i+1}:")
                            st.write(f"  Product: {emb['title']}")
                            st.write(f"  Product ID: {emb['product_id']}")
                            st.write(f"  Link: {emb['link']}")
                            st.write(f"  Time range: {emb['start_offset_sec']} - {emb['end_offset_sec']} seconds")
                            st.write(f"  Embedding vector (first 5 values): {emb['embedding'][:5]}")

            except Exception as e:
                st.error(f"Error with product {idx}: {str(e)}")

            # Add delay between videos
            if idx < total_products:
                with st.spinner('Waiting before processing next video...'):
                    time.sleep(5)

    # Final summary
    st.write("\n=== Final Summary ===")
    st.write(f"Total products processed: {total_products}")
    st.write(f"Total embeddings generated: {len(all_embeddings)}")
    return all_embeddings

# Add this to your Streamlit UI
st.title("Video Embedding Processor")

if st.button("Process All Products"):
    with st.spinner('Processing products...'):
        all_embeddings = process_products()
    st.success("Processing completed!")
