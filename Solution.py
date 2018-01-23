# Leetcode practice
# Han Wang
class Solution(object):
    def __init__(self):
        pass

    # 343 Integer break (PASSED)
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        a = [0,0,1,2,4,6,9]
        if n<7:
            return a[n]
        return max(self.integerBreak(n-2)*2, self.integerBreak(n-3)*3)

    # 371 Sum of two integers (PASSED 29ms)
    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        # consider bitwise operation
        return (a^b) + ((a&b) << 1)

    # 728 Self dividing numbers (PASSED 90ms)
    def selfDividingNumbers(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: List[int]
        """
        def isDividingNumber(n):
            for i in str(n):
                if i=='0':
                    return False
                elif n%int(i)>0:
                    return False
            return True
        ans = []
        for j in range(left, right+1):
            if isDividingNumber(j):
                ans+=[j]
        return ans

    # 766 Toeplitz matrix (PASSED 54ms)
    def isToeplitzMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: bool
        """
        # a M*N matrix has (m+n-1) diagonals
        a = range(-len(matrix)+1, len(matrix[0]))
        d = {}
        for i in a:
            d[i] = None

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if d[j-i]==None:
                    d[j-i] = matrix[i][j]
                elif matrix[i][j]!=d[j-i]:
                    return False
        return True

solution = Solution()
print(solution.selfDividingNumbers(1,58))