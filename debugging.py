# %% Case 1a - just vectors and matrices, discrete states/outcomes
array_path = os.path.join(os.getcwd(),'matlab_tests/spm_dot/randomCase1a_dot.mat')

mat_contents = loadmat(file_name=array_path)

A_array = mat_contents['A']
o_array = mat_contents['o']
s_array = np.array(mat_contents['s'],dtype=object)
result1_matlab = mat_contents['result1']
result2_matlab = mat_contents['result2']
result3_matlab = mat_contents['result3']

A = Categorical(values=A_array)

print(o_array.shape)
result1_python = A.dot(o_array, return_numpy = True)
print(np.isclose(result1_matlab,result1_python).all()==True)

# %%
result2_python = A.dot(s_array, return_numpy = True)
print(np.isclose(result2_matlab,result2_python).all()==True)

result3_python = A.dot(s_array, dims_to_omit = [1], return_numpy = True)
print(np.isclose(result3_matlab,result3_python).all()==True)

# %%
