from main import cosine_similarity 
import pytest 
def test_identical_vectors():
    vector=[1,2,3]
    similarity=cosine_similarity(vector,vector) 
    assert similarity == pytest.approx(1.0)
def test_orthogonal_vectors(): 
    vector_a = [1,0]
    vector_b=[0,1]
    similarity=cosine_similarity(vector_a,vector_b)
    assert similarity == pytest.approx(0.0)
def test_opposite_vectors():
    vector_a = [1,0]
    vector_b = [-1,0]
    similarity=cosine_similarity(vector_a,vector_b)
    assert similarity == pytest.approx(-1.0)
def test_zero_vector():
    vector_a = [0,0]
    vector_b = [1,1]
    similarity=cosine_similarity(vector_a,vector_b)
    assert similarity == pytest.approx(0.0)