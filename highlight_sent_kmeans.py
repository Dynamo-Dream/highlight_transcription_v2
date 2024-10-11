from youtube_transcript_api import YouTubeTranscriptApi
import re
from typing import List,Dict,Any
from fastapi import FastAPI, HTTPException
import uvicorn
from scipy.spatial.distance import euclidean
import spacy
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import numpy as np

app = FastAPI()
@app.get("/")
def hello():
    return {"result":"Hello I am Working"}
def spacy_tokenize(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def extract_video_id(url: str) -> str:
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    raise ValueError("Invalid YouTube URL")

def get_transcription(video_id: str) -> List[Dict[str, Any]]:
    try:
        return YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Unable to fetch transcript: {str(e)}")

def get_sentence_embedding(sentences):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return embeddings

def cluster_sentences(embeddings, ratio=0.3):
    # Determine number of clusters based on the ratio
    num_sentences = len(embeddings)
    num_clusters = max(1, int(ratio * num_sentences))
    
    # Fit K-means to cluster sentences
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    
    return kmeans

def select_representative_sentences(kmeans, sentences, embeddings):
    cluster_centers = kmeans.cluster_centers_
    selected_sentences = []
    
    for i, center in enumerate(cluster_centers):
        closest_idx = None
        closest_dist = float('inf')
        
        # Find the sentence closest to this cluster center
        for j, embedding in enumerate(embeddings):
            dist = euclidean(center, embedding)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = j
        
        selected_sentences.append(sentences[closest_idx])
    
    return selected_sentences
def get_result(summary: str, transcription: List[Dict[str, Any]], is_video_id: bool = True):
    offset_key = "start" if is_video_id else "offset"
    for data in transcription:
        data["text"] = data["text"].replace('&gt;', '').replace('\n', '').strip()
    result = []
    for data in transcription:
        if data["text"] in summary:
            result.append(data)
    return result

@app.post("/highlight")
def highlight_transcript(video_url:str):
    video_id = extract_video_id(video_url)
    transcription = get_transcription(video_id)
    text = " ".join([t["text"] for t in transcription])
    cleaned_text = text.replace('&gt;', '').replace('\n', '').strip()
    sentence = spacy_tokenize(cleaned_text)
    embeddings = get_sentence_embedding(sentence)
    kmeans = cluster_sentences(embeddings)
    summary = select_representative_sentences(kmeans,sentence,embeddings)
    summary = " ".join(summary)
    result = get_result(summary,transcription)
    return result

if __name__ == "__main__":
   uvicorn.run(app, host="127.0.0.1", port=8000)
    