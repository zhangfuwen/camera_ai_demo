from openai import OpenAI
import base64

def gemini_video_test(video_path, question, model="gemini-3-pro-preview"):
    """视频理解函数"""
    client = OpenAI(
        api_key="sk-EplRjGkWQ9CwXK5w10936448E56a46BcB487EeE809C6Bd40",  # 替换为您的 API Key
        base_url="https://api.apiyi.com/v1"
    )

    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()
        video_url = f"data:video/mp4;base64,{video_b64}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": video_url
                        },
                        "mime_type": "video/mp4",
                    }
                ]
            }
        ],
        temperature=0.2,
        max_tokens=4096
    )

    return response.choices[0].message.content
if __name__ == "__main__":
    video_path = "./file_example_MP4_480_1_5MG.mp4"  # 本地视频文件路径
    question = "请briefly描述这个视频的内容"

    result = gemini_video_test(video_path, question)
    print(result)
