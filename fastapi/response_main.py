import uvicorn
from fastapi import FastAPI

app = FastAPI()

#测试1：路径处理函数直接返回简单数据类型(字符串、数值、布尔)
#结果：返回简单数据类型会直接作为响应体响应到浏览器端，其中数据类型会自动转换
@app.get("/response/str")
def test_response_str():
    return "hello world"

#测试2：路径处理函数返回列表或字典
#结果：路径处理函数返回列表或字典会被转换为json（数组或对象），再作为响应体响应到浏览器端
@app.get("/response/dict")
def test_response_dict():
    # return {"name":"张三", "age":18, "gender": "male"}
    return ["张三","李四","王五"]

if __name__ == "__main__":
    uvicorn.run("response_main", host="127.0.0.8000", port=8000,reload=True)