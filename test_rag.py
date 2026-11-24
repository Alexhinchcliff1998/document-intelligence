from rag import PolicyIndex

def test_damage():
    result=PolicyIndex().answer('When must damage reports be submitted?')
    assert '48 hours' in result['answer']
    assert result['evidence'][0]['source'].startswith('claims:')

def test_vendor():
    assert '90 percent' in PolicyIndex().answer('vendor on-time delivery rate')['answer']

def test_no_evidence():
    assert not PolicyIndex().answer('neptune astronomy telescope')['evidence']
