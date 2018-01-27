# Leetcode practice
# Han Wang

from functools import reduce


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def __init__(self):
        pass

    # 17 Letter combintion of phone numbers (PASSED)
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        # dict
        if digits=='':
            return []
        d = {
            '2':'abc',
            '3':'def',
            '4':'ghi',
            '5':'jkl',
            '6':'mno',
            '7':'pqrs',
            '8':'tuv',
            '9':'wxyz'
        }
        return reduce(lambda a, digits: [x+y for x in a for y in d[digits]], digits, [''])

    # 38 Count and say (PASSED)
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        ans = '1'
        for _ in range(n - 1):
            t = None
            c = 0
            res = ''
            for j in range(len(ans) + 1):
                print(res)
                if j == len(ans):
                    res += str(c) + t
                elif not t:
                    t = ans[j]
                    c = 1
                elif t == ans[j]:
                    c += 1
                else:
                    res += str(c) + t
                    t = ans[j]
                    c = 1
            ans = res

        return ans

    # 167. Two Sum II - Input array is sorted (PASSED)
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        a, b = 0, len(numbers) - 1
        while (numbers[a] + numbers[b] != target):
            if numbers[a] + numbers[b] < target:
                a += 1
            else:
                b -= 1
        return a + 1, b + 1

    # 220 Contain duplicates III (PASSED)
    # Given an array of integers, find out whether
    # there are two distinct indices i and j in the
    # array such that the absolute difference between
    # nums[i] and nums[j] is at most t and the absolute
    # difference between i and j is at most k.
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        """
        # bucket sort
        if t<0:
            return False
        bucket = {}
        w = t+1

        for i in range(len(nums)):
            m = nums[i]/w
            if m in bucket:
                return True
            if m-1 in bucket and abs(bucket[m-1]-nums[i]) < w:
                return True
            if m+1 in bucket and abs(bucket[m+1]+nums[i]) < w:
                return True
            bucket[m] = nums[i]
            if i>k:
                del bucket[nums[i-k]/w]
        return False



    # 283 Move zeros
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        zero = 0  # records the position of "0"
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[zero] = nums[zero], nums[i]
                zero += 1

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

    # 378 Kth Smallest Element in a Sorted Matrix (PASS 71ms)
    def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        # this is actually a brute force solution but with good time cost
        return sorted(a for row in matrix for a in row)[k - 1]

    # 504 Base 7 (PASSED)
    def convertToBase7(self, num):
        """
        :type num: int
        :rtype: str
        """
        res = ''
        if num<0:
            res+='-'
            num=-num
        a = ''
        while num>6:
            a+=str(num%7)
            num = int(num/7)
        return res+(a+str(num))[::-1]

    # 628 Maximum product of three numbers (PASSED 117ms)
    def maximumProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        a = sorted(nums)
        return max(a[0]*a[1]*a[-1], a[-3]*a[-2]*a[-1])

    # 696 Count binary strings (PASSED)
    def countBinarySubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        s = map(len, s.replace('10', '1 0').replace('01', '0 1').split())
        return sum([min(a, b) for a, b in zip(s, s[1:])])

    # 724 Find pivot index (PASSED)
    def pivotIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if nums == []:
            return -1
        left = 0
        right = sum(nums) - nums[0]
        for i in range(len(nums)):
            if left == right:
                return i
            if i == len(nums) - 1:
                return -1
            left += nums[i]
            right -= nums[i + 1]
        return -1

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

    # 747. Largest Number At Least Twice of Others (PASSED)
    def dominantIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        a = sorted(nums)
        if len(nums) == 1:
            return 0
        if a[-1] >= 2 * a[-2]:
            return nums.index(a[-1])
        else:
            return -1

    # 760. Find Anagram Mappings (PASSED)
    def anagramMappings(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: List[int]
        """
        d = {}
        for i in range(len(B)):
            d[B[i]] = i
        return [d[A[i]] for i in range(len(A))]

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
print(solution.countAndSay(10))