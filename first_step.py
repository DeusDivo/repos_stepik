
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