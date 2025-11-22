# To install: pip install tavily-python
from tavily import TavilyClient
client = TavilyClient("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")
response = client.search(
    query="英伟达最新的显卡"
)
print(response)