from pymilvus import connections, utility

# 请确保这里的连接信息与你的实际配置一致
connections.connect(host='127.0.0.1', port='19530')

collection_name = 'kb_item_names'
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"✅ 已删除集合: {collection_name}")
else:
    print(f"ℹ️ 集合 {collection_name} 不存在")