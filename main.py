import os
import httpx
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# 1. Setup & Environment
load_dotenv()

app = FastAPI()

# CORS: Allows your index.html to communicate with this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("GEMINI_API_KEY")
# Using Gemini 2.5 Flash for speed
BASE_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"

class TopicRequest(BaseModel):
    topic: str

class VerifyRequest(BaseModel):
    question: str
    selected_option: str
    correct_answer: str

# 2. Robust Communication Function
async def call_gemini(prompt: str):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing from .env")

    url = f"{BASE_URL}?key={API_KEY}"
    # Removed generationConfig to avoid the "response_mime_type" error
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=45.0)
            res_data = response.json()

            if response.status_code != 200:
                error_msg = res_data.get("error", {}).get("message", "Gemini API Error")
                print(f"!!! Google API Error: {error_msg}")
                raise HTTPException(status_code=response.status_code, detail=error_msg)
            
            if "candidates" not in res_data or not res_data["candidates"]:
                print(f"!!! Safety Block or Empty Result: {res_data}")
                raise HTTPException(status_code=500, detail="AI Safety Filter blocked this topic.")
            
            return res_data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"!!! Request Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# 3. Agentic Content Generation
@app.post("/generate")
async def generate_content(request: TopicRequest):
    # This prompt tells the agent to use specific markers so we can parse the text easily
    agent_prompt = f"""
    You are the ClearLearn Agent. Create a micro-lesson on {request.topic} for ADHD students.
    
    INSTRUCTIONS:
    1. ANALYZE the topic.
    2. DRAFT a 4-step lesson with emojis and bolding.
    3. REFLECT on scan-ability and simplify.
    
    Format your response EXACTLY like this:
    LESSON_START
    (Lesson text here)
    LESSON_END
    QUIZ_JSON
    [
      {{"question": "Q1", "options": ["A", "B", "C", "D"], "answer": "correct string"}},
      {{"question": "Q2", "options": ["A", "B", "C", "D"], "answer": "correct string"}},
      {{"question": "Q3", "options": ["True", "False"], "answer": "True"}}
    ]
    """
    
    try:
        raw_text = await call_gemini(agent_prompt)
        
        # Extracting the Lesson using markers
        lesson_match = re.search(r"LESSON_START(.*?)LESSON_END", raw_text, re.DOTALL)
        lesson_part = lesson_match.group(1).strip() if lesson_match else "Lesson generation failed."
        
        # Extracting the JSON using markers
        quiz_match = re.search(r"QUIZ_JSON(.*)", raw_text, re.DOTALL)
        quiz_json_string = quiz_match.group(1).strip() if quiz_match else "[]"
        
        # Clean up Markdown formatting if the AI added backticks
        quiz_json_string = re.sub(r"```json|```", "", quiz_json_string).strip()
        
        return {
            "lesson": lesson_part,
            "quiz": json.loads(quiz_json_string)
        }
    except Exception as e:
        print(f"!!! Parsing Error: {e}\nRaw Output: {raw_text}")
        raise HTTPException(status_code=500, detail="Agent failed to format the response correctly.")

# 4. Agentic Verification
@app.post("/verify")
async def verify_answer(request: VerifyRequest):
    is_correct = request.selected_option.strip() == request.correct_answer.strip()
    
    verify_prompt = f"""
    The student picked "{request.selected_option}" for the question: "{request.question}".
    The correct answer is "{request.correct_answer}".
    
    Give a 1-sentence supportive explanation of why it is correct. Use an emoji.
    """
    
    explanation = await call_gemini(verify_prompt)
    return {"is_correct": is_correct, "explanation": explanation}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)