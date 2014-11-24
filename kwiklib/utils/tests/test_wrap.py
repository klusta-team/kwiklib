"""Unit tests for the wrap module."""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from kwiklib.utils import wrap, wrap_pd


# -----------------------------------------------------------------------------
# Wrap tests
# -----------------------------------------------------------------------------
def test_wrap_1():
    d = dict(key1=1, key2=2)
    dw = wrap(d)
    
    assert dw.key1 == 1
    assert dw.key2 == 2
    
def test_wrap_2():
    d = dict(key1=[0, 1], key2=[0, 10])
    dw = wrap(d)
    
    assert dw.key1 == [0, 1]
    assert dw.key2 == [0, 10]
    
def test_wrap_3():
    d = dict(key1=[0, 1], key2=dict(skey1=['0', '1'], skey2=['0', '10']))
    dw = wrap(d)
    
    assert dw.key1 == [0, 1]
    
    assert dw.key2[0].skey1 == '0'
    assert dw.key2[0].skey2 == '0'
    assert dw.key2[1].skey1 == '1'
    assert dw.key2[1].skey2 == '10'
    
    assert dw.key2.skey1 == ['0', '1']
    assert dw.key2.skey2 == ['0', '10']
    
def test_wrap_4():
    d = {'key1': 
            [
                {
                    'skey1': [0, 1], 
                    'skey2': [0, 10]
                },
                
                {
                    'skey1': [10, 11], 
                    'skey2': [10, 20]
                },   
            ]
        }
    dw = wrap(d)
    
    assert dw.key1[0].skey1 == [0, 1]
    assert dw.key1[0].skey2 == [0, 10]
    
    assert dw.key1[1].skey1 == [10, 11]
    assert dw.key1[1].skey2 == [10, 20]
    
def test_wrap_5():
    d = {'key1': 
            [
                {
                    'skey1': [
                                {'a': 1, 'b': 2},
                                {'a': 3, 'b': 4},
                             ], 
                    'skey2': [
                                {'a': 10, 'b': 20},
                                {'a': 30, 'b': 40},
                             ], 
                },
                
                {
                    'skey1': [
                                {'a': 5, 'b': 6},
                                {'a': 7, 'b': 8},
                             ], 
                    'skey2': [
                                {'a': 50, 'b': 60},
                                {'a': 70, 'b': 80},
                             ], 
                },   
            ]
        }
    dw = wrap(d)
    assert dw.key1[0].skey1.a[0] == 1
    assert dw.key1[0].skey1.a[1] == 3
    assert dw.key1[0].skey1.b[0] == 2
    assert dw.key1[0].skey1.b[1] == 4
    
    assert dw.key1.skey2[0].a[0] == 10
    assert dw.key1.skey2[0].a[1] == 30
    
def test_wrap_pd_1():
    d = [{'key1': {'index': i}} for i in range(100)]
    dw = wrap_pd(d)
    
    assert dw.key1.index[10] == 10
    assert np.array_equal(dw.key1.index[::2].values, np.arange(0, 100, 2))
    assert np.array_equal(dw.key1.index.values, np.arange(0, 100, 1))
    
    
    