# Calculations for the output size of the convolutional and maxpool layers

def calcconv(h,w,p,k,s):                                # height, width, padding, kernel size, stride
    output_h = (h + 2*p - k)//s + 1
    output_w = (w + 2*p - k)//s + 1
    
    print(output_h,output_w)

def calc_maxpool(h, w, k, s):                           # height, width, kernel size, stride
    output_h = (h - k) // s + 1
    output_w = (w - k) // s + 1
    
    print(output_h, output_w)
    
calcconv(291, 386,1,2,2)
calc_maxpool(72,96,2,2)