import numpy as np

def longestCommonSubsequence(s1: str, s2: str) -> int:
    n1 = len(s1)
    n2 = len(s2)
    if n1 == 0 or n2 == 0:
        return 0
    V = np.zeros((n1, n2), dtype=int)
    
    for i in range(n2):
        if s1[0] == s2[i]:
            V[0, i:] = 1
            break
    
    for j in range(n1):
        if s2[0] == s1[j]:
            V[j:, 0] = 1
            break
    
    for i in range(1, n1):
        for j in range(1, n2):
            V[i, j] = max(V[i-1, j], V[i, j-1], V[i-1, j-1] + int(s1[i]==s2[j]))
    return V[-1, -1]


s1 = 'abcde'
s2 = 'ace'
sol = longestCommonSubsequence(s1, s2)
print(sol)