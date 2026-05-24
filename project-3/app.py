import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def run_recommendation_engine():
    print("====================================================")
    print("   DecodeLabs AI Pipeline: Project 3 - Rec Logic    ")
    print("====================================================\n")

 
    items = [
        {"id": 1, "title": "Advanced Python Operations", "tags": "python programming backend automation software development"},
        {"id": 2, "title": "Deep Learning & Neural Networks", "tags": "ai machine learning artificial intelligence neural networks tensors deep learning"},
        {"id": 3, "title": "Full-Stack Web Architectures", "tags": "web development frontend backend react nodejs mern fullstack"},
        {"id": 4, "title": "MLOps & Model Deployment", "tags": "mlops machine learning cloud aws automation deployment production"},
        {"id": 5, "title": "Agentic Workflows with CrewAI", "tags": "ai artificial intelligence agents langchain crewai automation logic"}
    ]

    print("Enter your professional interests or tech stack keywords.")
    print("Example: 'ai development automation python'")
    print("---------------------------------------------------")
    user_input = input("Your Interests: ")
    
    user_profile = user_input.lower().strip()
    
    if not user_profile:
        print("System Error: Inputs cannot be empty. Terminating pipeline.")
        return

    print("\n[1/3] Processing text profiles into mathematical Vector Space...")
    
    item_descriptions = [item["tags"] for item in items]
    
    all_documents = [user_profile] + item_descriptions


    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_documents)

    user_vector = tfidf_matrix[0]
    item_vectors = tfidf_matrix[1:]

    print("[2/3] Computing Cosine Similarity metrics between User and Items...")
    

    similarity_scores = cosine_similarity(user_vector, item_vectors).flatten()

    print("[3/3] Ranking recommendations based on alignment scores...\n")

    ranked_indices = np.argsort(similarity_scores)[::-1]

    print("================= RECOMMENDATIONS =================")
    has_recommendation = False
    
    for idx in ranked_indices:
        score = similarity_scores[idx]
        if score > 0:
            item = items[idx]
            print(f"🎯 Rank: {item['title']}")
            print(f"   Score Alignment: {score * 100:.2f}%")
            print(f"   Matched Dimensions: {item['tags']}\n")
            has_recommendation = True
            
    if not has_recommendation:
        print("⚠️ No direct alignment found for your profile.")
        print("Suggestion: Try generic engineering keywords like 'python', 'ai', or 'web'.")
    print("===================================================\n")

if __name__ == "__main__":
    run_recommendation_engine()