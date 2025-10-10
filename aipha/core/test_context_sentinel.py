import yaml
from context_sentinel import ContextSentinel

def test_context_sentinel():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    sentinel = ContextSentinel(config)
    
    # Test add and get knowledge
    entry_id = sentinel.add_knowledge_entry("test_cat", "Test Title", "Test Content", {"meta": "value"})
    entries = sentinel.get_knowledge_entries("test_cat")
    assert len(entries) > 0
    assert "Test Content" in entries[0]['content']
    
    # Test verify integrity
    assert sentinel.verify_knowledge_base_integrity()
    
    # Test code example
    sentinel.add_code_example("Test Code Title", "print('Hello')", "Test explanation")
    code_results = sentinel.search_code_examples("Hello")
    assert len(code_results) > 0

if __name__ == "__main__":
    test_context_sentinel()
    print("ContextSentinel integration test passed.")