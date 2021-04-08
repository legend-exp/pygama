import pytest
import pygama.lgdo as lgdo 
from numpy.testing import assert_

@pytest.fixture()
def struct():
  print('--------setup--------')
  yield lgdo.Struct()
  print('--------tear down--------') 

@pytest.fixture()
def scalar():
  print('--------setup--------')
  yield lgdo.Scalar
  

@pytest.fixture()
def array():
  print('--------setup--------')
  yield lgdo.Array

class Test_Struct:
  def test_datatype_name(self,struct):
    result=struct.datatype_name()
    desired='struct'
    assert_(result==desired)
  
  def test_form_datatype(self,struct):
    result=struct.form_datatype()
    #verify
    desired='struct{}'
    assert_(desired==result)

  def test_add_field(self,struct,scalar, array):
    #set up,add scalar object 
    obj=scalar(value=10)
    name='scalar1'
    struct.add_field(name,obj)

    #verify 'struct{scl1}' is in attributes 
    desired_attr='struct{scalar1}'
    result_attr=struct.attrs['datatype']

    #and the correct type
    desired_type='Scalar'
    result_type=struct[name].__class__.__name__

    assert_(result_attr==desired_attr, f'Error with {desired_attr}: got {result_attr}')
    assert_(result_type==desired_type, f'got {result_type}')

    #add array and test updated attributes
    arr_obj=array(shape=(700,21), dtype='f', fill_val=2)
    name2='array1'
    struct.add_field(name2, arr_obj)
    expected='struct{scalar1,array1}'
    got=struct.attrs['datatype']
    
    #verify
    assert_(got==expected, f'Got {got}')

