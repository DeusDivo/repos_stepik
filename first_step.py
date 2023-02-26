#1523. Count Odd Numbers in an Interval Range
class Solution(object):
    def countOdds():
        print ("1523. Count Odd Numbers in an Interval Range")
        low = 2
        high = 7
        if low%2==0 and high%2==0:
            return high//2 - low//2
        else:
            return (high-low)//2+1
    print(countOdds())
##1491. Average Salary Excluding the Minimum and Maximum Salary
    def average():
        print("1491. Average Salary Excluding the Minimum and Maximum Salary")
        salary = [4000,3000,1000,2000]
        salary.remove(max(salary))
        salary.remove(min(salary))
        return sum(salary)/len(salary)
    print(average())
#191. Number of 1 Bits
    def gorti():
        print ("191. Number of 1 Bits")
        n = 11111111111111111111111111111101
        count = 0
        while n:
                n &= n - 1
                count += 1
        return count
    print(gorti())
        ##print(bin(n).count("1"))
#1281 subtractProductAndSum
    def subtractProductAndSum():
        print ("1281 subtractProductAndSum")
        n1 = 234
        l1 = [int(i) for i in str(n1)]
        product = 1
        sum1 = 0
        for i in l1:
            product *= i
            sum1 += i
        return (product-sum1)
    print(subtractProductAndSum())
#976. Largest Perimeter Triangle
    def largestPerimeter():
        nums = [2,1,2]
        print("976. Largest Perimeter Triangle")
        nums.sort()
        for i in range(len(nums)-2):
            if nums[i] + nums[i+1] > nums[i+2]:
                return nums[i] + nums[i+1] + nums[i+2]
        return 0
    print(largestPerimeter())
#1779. Find Nearest Point That Has the Same X or Y Coordinate
    def closestPoint():
        print("1779. Find Nearest Point That Has the Same X or Y Coordinate")
        #x = 3 ; y = 4 ; points = [[1,2],[3,1],[2,4],[2,3],[4,4]]
        #x = 3 ; y = 4 ; points = [[3,4]]
        #x = 3; y =4; points = [[2,3]]
        x = 5; y = 1; points = [[1,1],[6,2],[1,5],[3,1]]
        """
        min_dist = float('inf')
        min_index = -1
        valid = []
        for i in points:
            if i[0]==x or i[1] == y:
                valid.append(i)
                if len(valid) == 0 :
                    print (-1)
                else:
                    for i in range(len(valid)):
                        dist = abs(x - points[i][0]) + abs(y - points[i][1])
                        if dist < min_dist:
                            min_dist = dist
                            min_index = i
        """
        valid=[]
        for i in points:
            if i[0]==x or i[1]==y:
                valid.append(i)
        dist=[]
        if len(valid) == 0:
            return -1
        else :
            for i in valid:
                dist.append(abs(x - i[0]) + abs(y - i[1]))
            if valid[dist.index(min(dist))] in valid:
                #return points.index(valid[dist.index(min(dist))])
                return points.index(valid[dist.index(min(dist))])
    print(closestPoint())
#1822. Sign of the Product of an Array
    print("1822. Sign of the Product of an Array")
    def signFunc(x):
        if x>0:
            return 1
        elif x<0:
            return -1
        else:
            return 0
    nums = [1,5,0,2,-3] #input_1: -1,-2,-3,-4,3,2,1 input_3: -1,1,-1,1,-1
    product = 1
    for num in nums:
        product *= num
    print(signFunc(product))
#1502. Can Make Arithmetic Progression From Sequence
    def canFormAP():
        print("1502. Can Make Arithmetic Progression From Sequence")
        arr = [3,5,1]#input: 3,5,1
        arr.sort()
        diff = arr[1] - arr[0]

        for i in range(2, len(arr)):
            if arr[i] - arr[i-1] != diff:
                return False
            return True
    print(canFormAP())
#202. Happy Number
    def is_happy(n):
        print("202. Happy Number")
        seen = set()
        while n not in seen:
            seen.add(n)
            n = sum([int(i)**2 for i in str(n)])
        return n == 1
    print(is_happy(2))
#1790. Check if One String Swap Can Make Strings Equal
    def one_swap():
        diff = 0
        print("1790. Check if One String Swap Can Make Strings Equal")
        s1 = 'attack'; s2 = 'defend'#input_s1: attack , kelb : input_s2: defend, kelb
        """"
        for i in range (len(s1)):
            if s1[i]!=s2[i]:
                diff +=1
        if diff ==2:
            for i in range(len(s1)):
                if s1[i]!= s2[i]:
                        temp = list(s1)
                        temp[i] = s2[i]
                if ''.join(temp) == s2:
                     return True
        return False
        """
        if s1 == s2:
            return True
        elif sorted(s1) != sorted(s2):
            return False

        count =0
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                count += 1

        if count !=2:
            return False
        return True
    print(one_swap())
#1791. Find Center of Star Graph
    def find_center():
        print("1791. Find Center of Star Graph")
        edges =[[1,2],[2,3],[4,2]]# [1,2],[5,1],[1,3],[1,4]
        return (set(edges[0]) & set(edges[1])).pop()
    """
        degress ={}
        for u,v in edges:
            if u not in degress:
                degress[u] =0
            if v not in degress:
                degress[v] =0
            degress[u] += 1
            degress[v] += 1
        for node,degree, in degress.items():
            if degree == n-1:
                return node
        """
    print(find_center())
#
    def maxAverage():

        classes = [[1,2],[3,5],[2,2]]
        extraStudents = 2
        classes.sort(key=lambda x: x[0]/x[1])
        total_pass = 0
        total_students = 0
        for class_ in classes:
            total_pass += class_[0]
            total_students += class_[1]

        max_avg = total_pass / total_students

        for i in range(extraStudents):
            if classes[i][1] == classes[i][0]:  # no room to add extra student
                continue

            new_total = classes[i][1] + 1  # add one student to the class
            new_pass = classes[i][0] + 1  # one more student passed the exam

            avg = (total_pass - classes[i][0] + new_pass) / (total_students - classes[i][1] + new_total)

            max_avg = max(max_avg, avg)

        return max_avg
        """
        classes.sort(key=lambda x:x[0]/x[1], reverse=True)
        total_pass = 0
        total_students =0
        for passi, totali in classes:
            total_pass +=passi
            total_students += totali
        if extraStudents ==0:
            return total_pass/total_students
        for i in range(len(classes)):
            if extraStudents == 0: break
            passi, totali = classes[i]
            if (totali - passi) <= extraStudents:
                extraStudents-=(totali-passi)
                classes[i][0]=totali
            else:
                classes[i][0]+= extraStudents
                extraStudents =0
        new_total_pass =0
        for passi, _ in classes:
            new_total_pass+= passi
        return new_total_pass/total_students
        """
    print(maxAverage())
#Not rd 589. N-ary Tree Preorder Traversal
class Node:
    print("not Rd")
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
    def preorder(self, root: 'Node') -> list[int]:
        root = [1,0,3,2,4,0,5,6]
        output = []
        self.traverse(root, output)
        return output
    def traverse(self, root, output):
        if root is None: return
        output.append(root.val)
        for child in root.children:
            self.traverse(child, output)
    """
    def preorder (root):
        result = []
        if root is None:
            return result
        result.append(root.val)
        for child in root.children:
            result += preorder(child)
        return result
    """
    #496. Next Greater Element I
    def nextGreater():
        print("496. Next Greater Element I")
        nums1 = [2,4]; nums2 =[1,2,3,4] #input_1:2,4 input_2:1,2,3,4
        ans= []
        for i in range(len(nums1)):
            j = nums2.index(nums1[i])
            found = False
            for k in range(j+1, len(nums2)):
                if nums2[k]>nums1[i]:
                    ans.append(nums2[k])
                    found = True
                    break
            if not found:
                ans.append(-1)
        return ans
    print(nextGreater())
#1232. Check If It Is a Straight Line
    def chekStraight():
        print("1232. Check If It Is a Straight Line")
        coordinates = [[1,1],[2,2],[3,4],[4,5],[5,6],[7,7]]#input_1 [1,2],[2,3],[3,4],[4,5],[5,6],[6,7]

        if len(coordinates) == 2:
            return True

        x1,y1 = coordinates[0]
        x2,y2 = coordinates[1]

        for i in range(2, len(coordinates)):
            x,y = coordinates[i]

            if (y2-y1)*(x-x2)!=(y-y2)*(x2-x1):
                return False

        return True
    print(chekStraight())
#1588. Sum of All Odd Length Subarrays
    def sumODD():
        result = 0
        print ("1588. Sum of All Odd Length Subarrays")
        arr = [1,4,2,5,3]# 1,4,2,5,3
        for i in range(1, len(arr)+1, 2):
            for j in range(len(arr)-i+1):
                result += sum(arr[j:j+i])
        return result
    print (sumODD())
#283. Move Zeroes
    def moveZero():
        print("283. Move Zeroes")
        zero_count =0
        nums = [0,1,0,3,12]

        for i in range(len(nums)):
            if nums[i] !=0:
                nums[i-zero_count], nums[i] =nums[i], nums[i-zero_count]
            else:
                zero_count+=1
    print (moveZero())
#1672. Richest Customer Wealth
    def maxmumWealth():
        print("1672. Richest Customer Wealth")
        accounts = [[1,5],[7,3],[3,5]]# input_1: [1,2,3],[3,2,1]
        max_wealth = 0

        for row in accounts:
            wealth = sum(row)
            if wealth > max_wealth:
                max_wealth = wealth
        return max_wealth
    print (maxmumWealth())
#1572. Matrix Diagonal Sum
    def diagonalSum():
        print("1572. Matrix Diagonal Sum")
        mat = [[1,2,3],
              [4,5,6],
              [7,8,9]] #input_2: [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]
        n = len(mat)

        mid = n // 2

        summation = 0

        for i in range(n):

            # primary diagonal
            summation += mat[i][i]

            # secondary diagonal
            summation += mat[n-1-i][i]


        if n % 2 == 1:
            # remove center element (repeated) on odd side-length case
            summation -= mat[mid][mid]
        return summation
        """
        primary_diagonal = 0
        secondary_diagonal = 0

        for i in range(len(mat)):
            for j in range(len(mat[i])):
                if i == j:  # Primary diagonal
                    primary_diagonal += mat[i][j]

                if i + j == len(mat) - 1: # Secondary diagonal
                    secondary_diagonal += mat[i][j]

        return primary_diagonal + secondary_diagonal
        """
    print(diagonalSum())
#566. Reshape the Matrix
    def reshape():
        mat = [[1,2],[3,4]]
        r = 2
        c = 4
        print("566. Reshape the Matrix")
        if len(mat)*len(mat[0])!=r*c:
            return mat

        new_mat = [[0 for _ in range(c)] for _ in range(r)]

        row = 0
        col = 0

        for i in range(len(mat)):
            for j in range(len(mat[0])):
                new_mat[row][col] = mat[i][j]
                col+=1
                if col == c:
                    row +=1
                    col=0
        return new_mat
    print(reshape())
#1768. Merge Strings Alternately
    def merge_string():
        word1 = 'ab'
        word2 = 'pqrs'
        print("1768. Merge Strings Alternately")
        merged_string = ""

        for i in range(max(len(word1), len(word2))):
            if i < len(word1):
                merged_string += word1[i]
            if i < len(word2):
                merged_string += word2[i]
        return merged_string
    print(merge_string())
#1678. Goal Parser Interpretation
    def interpred():
        print("1678. Goal Parser Interpretation")
        command = 'G()(al)'
        #result = ""
        newstr = command.replace('()','o')
        return newstr.replace('(al)','al')
        """
        for char in command:
            if char == "G":
                result += "G"
            elif char == "()":
                result += "o"
            elif  char == "(al)":
                result += "al"
        return result
        """
    print(interpred())
#389. Find the Difference
    def addedLetter():
        print("389. Find the Difference")
        s = 'abcd'
        t = 'abcde'
        """
        counter_s = Counter(s)
        counter_t = Counter(t)
        counter_result = counter_t-counter_s
        for key,val in counter_result.items():
            if val==1:
                return key

        for i in range(len(s)):
            if s[i] not in t:
                return t[i]
        """
    print(addedLetter())
#1309. Decrypt String from Alphabet to Integer Mapping
    def freqAlfabets():
        print("1309. Decrypt String from Alphabet to Integer Mapping")
        s = '10#11#12#'#input_2 1326#
        result =""
        i =0
        while i< len(s):
            if i+2 <len(s) and s[i+2] =='#':
                result += chr(int(s[i:i+2])+96)
                i+=3
            else:
                result+= chr(int(s[i])+96)
                i+= 1
        return result
    print(freqAlfabets())
#953. Verifying an Alien Dictionary
    def isAleanSort():
        words = ["hello","leetcode"]
        order = "hlabcdefgijkmnopqrstuvwxyz"
        print ("953. Verifying an Alien Dictionary")
        map = {}
        for i in range(len(order)):
            map[order[i]] = i
        for i in range(1, len(words)):
            first = words[i - 1]
            second = words[i]
            n = min(len(first), len(second))
            flag = False
            for j in range(n):
                if map[first[j]] < map[second[j]]:
                    flag = True
                    break
                elif map[first[j]] > map[second[j]]:
                    return False
            if not flag and len(first) > len(second):
                return False
        return True
        """
        order_index = {c: i for i, c in enumerate(order)}

        for i in range(len(words)-1):
            word1 = words[i]
            word2 = words[i+1]

        for k in range(min(len(word1), len(word2))):
            if word1[k]!=word2[k]:
                if order_index[word1[k]]>order_index[word2[k]]:
                    return False
                break
            else:
                if len(word1)>len(word2):
                    return False
        return True
        """
    print(isAleanSort())
    print(isAleanSort())
#1290. Convert Binary Number in a Linked List to Integer
class ListNode:
    def __init__(self,x):
        self.val = x
        self.next = None
class solution_1290:
    def getDecima(self,head:ListNode) -> int:
        print("1290. Convert Binary Number in a Linked List to Integer")
        head = [1,0,1]
        decima_value = 0
        while head != None:
            decima_value = (decima_value<<1)| head.val
            head = head.next
        return decima_value
"""
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def middleNode(self):
        slow_ptr = self.head
        fast_ptr = self.head

        if self.head is not None:
            while (fast_ptr is not None and fast_ptr.next is not None):
                fast_ptr = fast_ptr.next.next
                slow_ptr = slow_ptr.next
            return slow_ptr.data
"""
#
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None
    def  findMiddleNode(head):
        slow = head
        fast = head

        while (fast is not None and fast.next is not None):
            slow = slow.next
            fast = fast.next
            return slow
        print(slow)
#
    def __init__(self) -> None:
        self.val = val
        self.left = left
        self.righ = right
    def maxDepth():
        if root is None:
            return
        else:
            lDepth = self.maxDepth(root.left)
            rDepth = self.maxDepth(root.righ)
        if (lDepth > rDepth):
            return lDepth+1
        else:
            return rDepth+1
    def sumofleft():
        if not root:
            return 0
            if root.left and not root.left.left and not root.left.right:
                return root.left.val +self.sumofleft(root.righ)
            return self.sumoleft(root.left)+ self.sumofleft(root.right)
 #
    def sumOfLeftLeaves():
        root = [3,9,20,null,null,15,7]
        s, ans = deque([(root, False)]), 0
        while s:
            cur, isLeft = s.pop()
            if not cur.left and not cur.right and isLeft:
                ans = ans + cur.val
            if cur.right:
                s.append((cur.right, False))
            if cur.left:
                s.append((cur.left, True))
        return ans
    def sortByBits():
        arr = [0,1,2,3,4,5,6,7,8]
        return sorted(arr, key = lambda x: (bin(x).count('1'), x))
    print(sortByBits())
#   242. Valid Anagram
    def isAnagram():
        print ("242. Valid Anagram")
        s = "anagram"
        t = "nagaram"
        s_list = list(s)
        t_list = list(t)

        s_list.sort()
        t_list.sort()

        if s_list == t_list:
            return True
        return False
    print(isAnagram())
#v217. Contains Duplicate
    def contains():
        print("217. Contains Duplicate")
        nums = [1,2,3,1]
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False
    print(contains())
#54. Spiral Matrix
    def spiralMatrix():
        matrix = [[1,2,3],[4,5,6],[7,8,9]]
        print("54. Spiral Matrix")

        result = []
        while matrix:
            result+= matrix.pop(0)
            matrix = list(zip(*matrix))[::-1]
        return result
    print(spiralMatrix())
#1706. Where Will the Ball Fall
    def solve():

        grid = [[1,1,1,-1,-1],[1,1,1,-1,-1],[-1,-1,-1,1,1],[1,1,1,1,-1],[-1,-1,-1,-1,-1]]
        n =5
        print ("1706. Where Will the Ball Fall")
        answer = [-1]*n

        for i in range(n):
            x,y =0, i
            while x <len(grid) and y <len(grid[0]):
                if grid[x][y] == 1:
                    y += 1
                elif grid[x][y] == -1:
                    y -= 1
                x+=1
            if y>=0 and y <len(grid[0]):
                answer[i] = y
            return answer
    print (solve())
#14 Longest Common Prefix
    def longest():
        strs = ["14 flower","flow","flight"]
        print('Longest Common Prefix')
        if not strs:
            return ""

        shortest_str = min(strs,key=len)

        for i, ch in enumerate(shortest_str):
            for other in strs:
                if other[i]!=ch:
                    return shortest_str[:i]
        return shortest_str
    print(longest())
#43. Multiply Strings
    def multiply():
        num1 = 2
        num2 = 3
        print("43. Multiply Strings")
        result = int(num1)*int(num2)
        return str(result)
    print(multiply())
#896. Monotonic Array
    def isMonotonic():
        nums = [1,2,2,3]
        print('896. Monotonic Array')
        increasing = decreasing = True
        for i in range(len(nums)-1):
            if nums[i]>nums[i+1]:
                increasing = False
            if nums[i]< nums[i+1]:
                decreasing = False
        return increasing or decreasing
    print(isMonotonic())
#28. Find the Index of the First Occurrence in a String
    def find_needle():
        haystack = "sadbutsad"
        needle = "sad"
        print('28. Find the Index of the First Occurrence in a String')
        for i in range(len(haystack)):
            if haystack[i:i+len(needle)]==needle:
                return i
        return -1
    print(find_needle())
#34. Find First and Last Position of Element in Sorted Array
    def find_first():
        print("34. Find First and Last Position of Element in Sorted Array")
        nums =[5,7,7,8,8,10]
        target = 8
        star_index = -1
        end_index = -1
        left = 0
        right = len(nums)-1

        while left <= right:
            mid = (left +right)//2
            if nums[mid] == target:
                star_index = mid
                right = mid - 1
            elif nums[mid]<target:
                left = mid +1
            else:
                right = mid - 1

        left = 0
        right = len(nums) -1

        while left <= right:
            mid = (left+right)//2

            if nums[mid] == target:
                end_index = mid
                left = mid +1

            elif nums[mid]<target:
                left = mid + 1
            else:
                right = mid -1
        return [star_index,end_index]
    print(find_first())
#33. Search in Rotated Sorted Array
    def search_rotate():
        print("33. Search in Rotated Sorted Arra")
        nums = [4,5,6,7,0,1,2]
        target = 0
        left = 0
        right = len(nums)-1

        while left <= right:
            mid = (left+right)//2
            if nums[mid] == target:
                return mid
            if nums[left] <= nums[mid]:
                if nums[left]<=target<nums[mid]:
                    right = mid - 1
                else:
                    left = mid +1
            else:
                if nums[mid]<target<=nums[right]:
                    left = mid-1
                else:
                    rigth = mid -1
        return -1
    print(search_rotate())
#74. Search a 2D Matrix
    def search_matrix():
        matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
        target = 3
        print ("74. Search a 2D Matrix")

        if not matrix or not matrix[0]:
            return False

        m,n = len(matrix), len(matrix[0])
        left,right = 0 , m*n-1

        while left <= right:
            mid = (left + right)//2
            row,col = mid//n, mid%n

            if matrix[row][col] == target:
                return True

            elif matrix[row][col] < target:
                left = mid + 1
            else:
                right = mid-1
        return False
    print(search_matrix())
#2468. Split Message Based on Limit
    def split_message():
        print('2468. Split Message Based on Limit')
        message = "this is really a very awesome message"
        limit = 9
        parts = []
        part_count = 0

        while message:
            part_count += 1
            if len(message)<= limit:
                parts.append(message + f"<{part_count}/{part_count}>")
                break
            else:
                part = message[:limit - len(str(part_count))-2]+f"<{part_count}/"
                message = message[limit-len(str(part_count))-2:]
                parts.append(part)
        return parts
    print(split_message())
#153. Find Minimum in Rotated Sorted Array
    def findMin():
        print('153. Find Minimum in Rotated Sorted Array')
        nums = [3,4,5,1,2]
        min = nums[0]

        for i in range(1,len(nums)):
            if nums[i]<min:
                min = nums[i]
        return min
    print(findMin())
#162. Find Peak Element
    def findpeak():
        print('162. Find Peak Element')
        nums = [1,2,3,1]

        for i in range(len(nums)):
            if (i == 0 or nums[i]>nums[i-1]) and (i == len(nums)-1 or nums[i]>nums[i+1]):
                return i
        return -1
    print(findpeak())
#989. Add to Array-Form of Integer
    def add_k():
        print('989. Add to Array-Form of Integer')
        nums =[1,2,0,0]
        k = 34

        result = []
        carry = 0

        for i in range(len(nums)-1,0,-1):
            curr_sum = nums[i] + k + carry
            result.append(curr_sum%10)
            carry  =curr_sum//10

        if carry > 0:
            result.append(carry)

        return result[::-1]

    print(add_k())
#67. Add Binary
    def add_binar():
        print("67. Add Binary")
        a = "11"
        b = "1"
        return (bin(a,2))
        """
        result = ""
        carry =0
        i = len(a)-1
        j = len(b)-1

        while i>=0 or j >=0:
            sum = carry
            if i>=0:
                sum +=int(a[i])
            if j >= 0:
                sum += int(b[j])

            result += str(sum%2)
            carry = sum//2

            i-=1
            j-+1
        if carry !=0:
            result +=str(carry)
        return result[::-1]
    """
    #print(add_binar())
#110. Balanced Binary Tree
#459. Repeated Substring Pattern
    def repeatedSub():
        print('459. Repeated Substring Pattern')
        s = "abab"
        n =len(s)

        for i in range (n//2):
            curr_len = i+1
            prev_str = s[0:curr_len]
            curr_str = s[curr_len:2*curr_len]

        while curr_str == prev_str and 2 * curr_len <=n:
            prev_str = curr_len
            curr_len = 2 *curr_len
            curr_len=s[curr_len:2*curr_len]
        if (2*curr_len == n and prev_str == curr_len):
            return True
        return False
    print(repeatedSub())
#150. Evaluate Reverse Polish Notation
    def evaluate_rpn():
        tokens = ["4","13","5","/","+"]
        print('150. Evaluate Reverse Polish Notation')
        stack = []

        for token in tokens:
            if token in ['+','-','*','/']:
                b = stack.pop()
                a = stack.pop()

                if token =='+':
                    stack.append(a+b)
                elif token == '-':
                    stack.append(a-b)
                elif token == '*':
                    stack.append(a*b)
                else:
                    token == '/'
                    stack.append(int(a/b))
            else:
                stack.append(int(token))
        return stack[0]
    print(evaluate_rpn())
#66. Plus One
    def incrementInteger():
        print('66. Plus One')
        carry = 1
        digits = [1,2,3]

        for i in range(len(digits)-1,-1,-1):
            digits[i] += carry
            if digits[i] == 10:
                digits[i] = 0
                carry = 1
            else:
                carry = 0

            if carry == 1:
                digits.insert(0,1)
            return digits
    print(incrementInteger())
#43. Multiply Strings
    def multiply():
        print('43. Multiply Strings')
        num1 = '2'
        num2 = '3'

        return str(int(num1)*int(num2))
    print(multiply())
#739. Daily Temperatures
    def dailyTemperatures():
        print("739. Daily Temperatures")
        temp = [73,74,75,71,69,72,76,73]
        answer = [0]*len(temp)
        stack =[]

        for i, t in enumerate(temp):
            while stack and temp[stack[-1]] < t:
                curr_i = stack.pop()
                answer[curr_i] = i - curr_i
            stack.append(i)
        return answer
    print(dailyTemperatures())
#58. Length of Last Word
    def last_words():
        print('58. Length of Last Word')
        s = "Hello word"
        word = s.split()
        s1 = len(word[-1])
        return s1
#48. Rotate Image
    def rotate_image():
        print('48. Rotate Image')
        matrix = [[1,2,3],[4,5,6],[7,8,9]]
        n = len(matrix)

        for i in range(n//2):
            for j in range(i, n -i - 1):
                temp = matrix[i][j]
                matrix[i][j] = matrix[n - j -1][i]
                matrix[n - j -1][i] = matrix[n - i -1][n - j - 1]
                matrix[n - i -1][n - j - 1] = matrix[j][n - i -1]
                matrix[j][n-i-1] = temp
                return matrix[i][j]
    print(rotate_image())
#1886. Determine Whether Matrix Can Be Obtained By Rotation
    def determine():
        print('1886. Determine Whether Matrix Can Be Obtained By Rotation')
        mat = [[0,1],[1,0]]
        target = [[1,0],[0,1]]
        n = len(mat)

        if n!= len(target):
            return False
        temp = [[0 for x in range(n)] for y in range(n)]

        for i in range(n):
            for j in range(n):
                temp[j][n-1-i] = mat[i][j]

        if temp == target:
            return True

        for i in range(n):
            for j in range(n):
                temp[n-1-j][i] = mat[i][j]

        if temp == target:
            return True

        return False
    print(determine())
#15 3Sum
    def threeSum():
        print('3Sum')
        nums = [-1,0,1,2,-1,-4]
        res =[]
        nums.sort()
        for i in range(len(nums)-2):
            if nums[i] + nums[i+1] +nums[i+2]>0:
                break
            if i >0 and nums[i] == nums[i-1]:
                continue
            l, r = i+1, len(nums)-1
            while l <r:
                s = nums[i]+nums[l]+nums[r]
                if s <0:
                    l+=1
                elif s >0:
                    r-+1
                else:
                    res.append((nums[i], nums[l], nums[r]))
                    while l < r and nums[l] == nums[l+1]:
                        l+=1
                    while l<r and nums[r] == nums[r-1]:
                        r-=1
                    l +=1; r -=1
        return set(res)
    print(threeSum())
#844. Backspace String Compare
    def backspaceCompare():
        print('844. Backspace String Compare')
        s = "ab#c"; t = "ad#c"
        s_stack = []
        t_stack = []

        for char in s:
            if char !='#':
                s_stack.append(char)
            elif s_stack:
                s_stack.pop()

        for char in t:
            if char !='#':
                t_stack.append(char)
            elif t_stack:
                t_stack.pop()
        return s_stack == t_stack
    print(backspaceCompare())
#986. Interval List Intersections
    def intervallntersection():
        print('986. Interval List Intersections')
        firstList = [[0,2],[5,10],[13,23],[24,25]]
        secondList = [[1,5],[8,12],[15,24],[25,26]]
        resulr = []
        i, j =0,0

        while i < len(firstList) and j<len(secondList):
            a,b = firstList[i]
            c,d = secondList[j]

            if b>= c and d>=a:
                start = max(a,c)
                end = min(b,d)
                resulr.append([start,end])
            if b<d:
                i+=1
            else:
                j+=1
        return resulr
    print(intervallntersection())
#973. K Closest Points to Origin
    def kCloset():
        print('973. K Closest Points to Origin')
        points = [[1,3],[-2,2]]
        k = 1
        points.sort(key = lambda P: P[0]**2 + P[1]**2)
        return points[k]
    print(kCloset())
#1630. Arithmetic Subarrays
    def is_airt():
        print('1630. Arithmetic Subarrays')
        nums = [4,6,5,9,3,7]
        l = [0,0,2]
        r = [2,3,5]
        res = []

        for i in range(len(l)):
            sub_arr = sorted(nums[l])