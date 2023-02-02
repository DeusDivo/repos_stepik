print ("1523. Count Odd Numbers in an Interval Range",'\n')
low = 8
high = 10
if low%2==0 and high%2==0:
    print (high//2 - low//2)
else:
    print ((high-low)//2+1)
