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

    # 190 Reverse bits (PASSED)
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        return int(bin(n)[2:].zfill(32)[::-1], base=2)

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

    # 326 Power of 3 (PASSED)
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        return n > 0 and (3 ** 21) % n == 0


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

    # 409 Longest Palindrome (PASSED 43ms)
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """
        d = {}
        for i in s:
            if i not in d:
                d[i] = 1
            else:
                d[i] += 1
        res = 0
        for i in d.keys():
            if d[i] % 2 == 0:
                res += d[i]
            else:
                res += d[i] - 1
        if res == len(s):
            return res
        else:
            return res + 1

    # 461 Hamming distance
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        return bin(x ^ y).count('1')

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

    # 506 Relative Ranks
    def findRelativeRanks(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        a = sorted(nums, reverse=True)
        res = []
        for i in nums:
            if i == a[0]:
                res += ['Gold Medal']
            elif i == a[1]:
                res += ['Silver Medal']
            elif i == a[2]:
                res += ['Bronze Medal']
            else:
                res += [str(a.index(i) + 1)]
        return res

    # 521 Longest Uncommon Subsequence I (PASSED)
    def findLUSlength(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: int
        """
        return -1 if a == b else max(len(a), len(b))

    # 581 Shortest Unsorted Continuous Subarray (PASSED)
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        a = []
        s_nums = sorted(nums)
        for i in range(len(nums)):
            if s_nums[i] != nums[i]:
                a += [i]
        return a[-1] - a[0] + 1 if a else 0

    # 628 Maximum product of three numbers (PASSED 117ms)
    def maximumProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        a = sorted(nums)
        return max(a[0]*a[1]*a[-1], a[-3]*a[-2]*a[-1])

    # 643 Maximum Average Subarray I (PASSED)
    def findMaxAverage(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: float
        """
        if k == len(nums):
            return float(sum(nums)) / k

        m = sum(nums[:k])
        tmp = m
        for i in range(len(nums) - k):
            tmp = tmp - nums[i] + nums[i + k]
            m = max(m, tmp)
        return float(m) / k

    # 696 Count binary strings (PASSED)
    def countBinarySubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        s = map(len, s.replace('10', '1 0').replace('01', '0 1').split())
        return sum([min(a, b) for a, b in zip(s, s[1:])])

    # 721. Accounts Merge (PASSED)
    def accountsMerge(self, accounts):
        """
        :type accounts: List[List[str]]
        :rtype: List[List[str]]
        """

        d = {}
        for i, a in enumerate(accounts):
            for j in a[1:]:
                if j not in d:
                    d[j] = []
                d[j] += [i]
        visited = [False]*len(accounts)

        def dfs(i, email):
            if visited[i]:
                return
            visited[i] = True

            for m in accounts[i][1:]:
                email.add(m)
                for n in d[m]:
                    dfs(n, email)

        res = []
        for i, a in enumerate(accounts):
            if visited[i]:
                continue
            n, email = a[0], set()
            dfs(i, email)
            res += [[n]+sorted(email)]
        return res

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
print(solution)