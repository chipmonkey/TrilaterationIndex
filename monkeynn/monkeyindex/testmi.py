import monkeyindex

x = monkeyindex.monkeyindex(5)
print(x.length)
print(x.mi)
print(x.mi['distance'])
print(x.mi['distance'][0].shape)
x.loadmi([1,2,3,4,5])
print(x.mi)
