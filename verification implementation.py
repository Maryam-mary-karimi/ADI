#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import binascii
import time
import hashlib
import hmac
import math 
import sympy
from functools import reduce
import numpy as np
import os
import base64

bftime=0
bfst=0
bf=[0] * 10700

treetime=0
treest=0
treecounter=0
hmacData=""
person=""

dpdptime=0
dpdpst=0

portime=0
porst=0


# In[2]:


def modInverse2(a, m):
    for x in range(1, m):
        if (((a%m) * (x%m)) % m == 1):
            return x
    return 1

def modInverse(a, m):
    g = gcd(a, m)
    if (g != 1):
        print("Inverse doesn't exist")
    else:
        # If a and m are relatively prime,
        # then modulo inverse is a^(m-2) mode m
        print("Modular multiplicative inverse is ",
              power(a, m - 2, m))
        return power(a, m - 2, m)

# To compute x^y under modulo m 
def power(x, y, m):
    if (y == 0):
        return 1
    p = power(x, y // 2, m) % m
    p = (p * p) % m 
    if(y % 2 == 0):
        return p
    else:
        return ((x * p) % m)
 
 
def gcd(a, b):
    if (a == 0):
        return b
    return gcd(b % a, a)
 

# Randomly select 2 primes with same Bitlength l/2
p = sympy.randprime(100000000, 10000000000)#(l/2)
q = sympy.randprime(100000000, 10000000000)#(l/2)
# Compute
n = p * q
phi = (p - 1) * (q - 1)
# Select an arbitrary integer e with 1 < e < phi and gcd(e,phi) == 1
e = int(sympy.randprime(1, phi))
# Compute the integer d statisfying 1 < d < phi and e*d == 1 % phi
d = modInverse(e, phi)
# Return n e d
print("Public Key: " + str(e))
print("Private Key: " + str(d))
print("n = " + str(n))  


# In[3]:


def bloomfilter(data):
    global bftime;
    start = time.time_ns()
    #print(start)
    tbf=[0]*20
    for i in ["1","2","3"]:
        result = hash((i+data))%20
        tbf[result]=1
    #print(tbf)    
    data=str(tbf)    
    #print(data)
    
    for i in ["","sec","thr","four"]:
        result = hash((i+data))%10700
        bf[result]=1
    for i in ["fifth","six","sev"]:    
        #result = hashlib.md5((i+data))%10700 
        result = hash((i+data))%10700
        bf[result]=0
    end = time.time_ns()
    #print(end)
    bftime=bftime+(end - start)
    
    #print(bftime)

bloomfilter("salam")


# In[ ]:





# In[4]:


def tree(data):
    global treecounter;
    global treetime;
    global treest;
    global hmacData;
    global e;
    global person;
    treecounter=treecounter+1;
   
    start = time.time_ns()
    
    tag = hashlib.sha256(data.encode())
    
    newperson=data.split(",")[0]
    if(person!=newperson):
        print(person)
        treecounter=1
        hmacData=""

        
    hmacData=hmacData+str(tag)
    #d(number of children 6)
    if(treecounter%6==0):
        print("parent")
        key_bytes= bytes(str(e) , 'latin-1') # Commonly 'latin-1' or 'utf-8'
        data_bytes = bytes(str(hmacData), 'latin-1') 
        digest = hmac.new(key_bytes, data_bytes , hashlib.sha256).hexdigest()
        hmacData=""
        treest=treest+len(digest)
    end = time.time_ns()
    
    treetime=treetime+(end - start)
    treest=treest+len(str(tag))  
    
    #print(treetime)
    #print(treest)
    
    # print(result.hexdigest()) 
    # result = hashlib.sha384(str.encode()) 
    # result = hashlib.sha224(str.encode()) 
    # result = hashlib.sha512(str.encode()) 
    # result = hashlib.sha1(str.encode()) 

tree("0,admission.ward:Cardiology")


# In[5]:


def dpdp(data):
    global dpdptime;
    global dpdpst;
    start = time.time_ns()
    #bin(int(binascii.hexlify('hello'), 16))
    #dtaAscii=''.join(str(ord(c)) for c in data)
    #dataAscii=reduce(lambda x, y: str(x)+str(y), map(ord,data))
    datascii = np.fromstring(data, dtype=np.uint8)
    oneAscii=""
    for c in datascii:
        oneAscii=oneAscii+str(c)
    #print(oneAscii)
    b = int(oneAscii,16)
    tag = pow (e, b , n)
    end = time.time_ns()
    #print(len(str(tag)))
    dpdptime=dpdptime+(end - start)
    dpdpst=dpdpst+len(str(tag))  
    
    #print(dpdptime)
    #print(dpdpst)
    
dpdp("salam")


# In[6]:


#https://en.wikiversity.org/wiki/Reed%E2%80%93Solomon_codes_for_coders#Encoding_main_function

def gf_mult_noLUT(x, y, prim=0):
    '''Multiplication in Galois Fields without using a precomputed look-up table (and thus it's slower)
    by using the standard carry-less multiplication + modular reduction using an irreducible prime polynomial'''

    ### Define bitwise carry-less operations as inner functions ###
    def cl_mult(x,y):
        '''Bitwise carry-less multiplication on integers'''
        z = 0
        i = 0
        while (y>>i) > 0:
            if y & (1<<i):
                z ^= x<<i
            i += 1
        return z

    def bit_length(n):
        '''Compute the position of the most significant bit (1) of an integer. Equivalent to int.bit_length()'''
        bits = 0
        while n >> bits: bits += 1
        return bits

    def cl_div(dividend, divisor=None):
        '''Bitwise carry-less long division on integers and returns the remainder'''
        # Compute the position of the most significant bit for each integers
        dl1 = bit_length(dividend)
        dl2 = bit_length(divisor)
        # If the dividend is smaller than the divisor, just exit
        if dl1 < dl2:
            return dividend
        # Else, align the most significant 1 of the divisor to the most significant 1 of the dividend (by shifting the divisor)
        for i in range(dl1-dl2,-1,-1):
            # Check that the dividend is divisible (useless for the first iteration but important for the next ones)
            if dividend & (1 << i+dl2-1):
                # If divisible, then shift the divisor to align the most significant bits and XOR (carry-less subtraction)
                dividend ^= divisor << i
        return dividend
    
    ### Main GF multiplication routine ###

    # Multiply the gf numbers
    result = cl_mult(x,y)
    # Then do a modular reduction (ie, remainder from the division) with an irreducible primitive polynomial so that it stays inside GF bounds
    if prim > 0:
        result = cl_div(result, prim)

    return result


gf_exp = [0] * 512 # Create list of 512 elements. In Python 2.6+, consider using bytearray
gf_log = [0] * 256

def init_tables(prim=0x11d):
    '''Precompute the logarithm and anti-log tables for faster computation later, using the provided primitive polynomial.'''
    # prim is the primitive (binary) polynomial. Since it's a polynomial in the binary sense,
    # it's only in fact a single galois field value between 0 and 255, and not a list of gf values.
    global gf_exp, gf_log
    gf_exp = [0] * 512 # anti-log (exponential) table
    gf_log = [0] * 256 # log table
    # For each possible value in the galois field 2^8, we will pre-compute the logarithm and anti-logarithm (exponential) of this value
    x = 1
    for i in range(0, 255):
        gf_exp[i] = x # compute anti-log for this value and store it in a table
        gf_log[x] = i # compute log at the same time
        x = gf_mult_noLUT(x, 2, prim)

        # If you use only generator==2 or a power of 2, you can use the following which is faster than gf_mult_noLUT():
        #x <<= 1 # multiply by 2 (change 1 by another number y to multiply by a power of 2^y)
        #if x & 0x100: # similar to x >= 256, but a lot faster (because 0x100 == 256)
            #x ^= prim # substract the primary polynomial to the current value (instead of 255, so that we get a unique set made of coprime numbers), this is the core of the tables generation

    # Optimization: double the size of the anti-log table so that we don't need to mod 255 to
    # stay inside the bounds (because we will mainly use this table for the multiplication of two GF numbers, no more).
    for i in range(255, 512):
        gf_exp[i] = gf_exp[i - 255]
    return [gf_log, gf_exp]

def gf_mul(x,y):
    #print("gf_mul")
    if x==0 or y==0:
        return 0
    return gf_exp[gf_log[x] + gf_log[y]] # should be gf_exp[(gf_log[x]+gf_log[y])%255] if gf_exp wasn't oversized

def gf_pow(x, power):
    #print("gf_pow ", x, " ", power, ", log ",gf_log[x], " , gf_exp ",gf_exp[(gf_log[x] * power) % 255])
    return gf_exp[(gf_log[x] * power) % 255]

def gf_inverse(x):
    #print("gf_inverse")
    return gf_exp[255 - gf_log[x]] # gf_inverse(x) == gf_div(1, x)


def gf_poly_mul(p,q):
    #print("gf_poly_mul")
    '''Multiply two polynomials, inside Galois Field'''
    # Pre-allocate the result array
    r = [0] * (len(p)+len(q)-1)
    # Compute the polynomial multiplication (just like the outer product of two vectors,
    # we multiply each coefficients of p with all coefficients of q)
    for j in range(0, len(q)):
        for i in range(0, len(p)):
            r[i+j] ^= gf_mul(p[i], q[j]) # equivalent to: r[i + j] = gf_add(r[i+j], gf_mul(p[i], q[j]))
                                                         # -- you can see it's your usual polynomial multiplication
    return r

def rs_generator_poly(nsym):
    #print("rs_generator_poly")
    '''Generate an irreducible generator polynomial (necessary to encode a message into Reed-Solomon)'''
    g = [1]
    for i in range(0, nsym):
        #print("g ",g)
        #print(gf_pow(2, i))
        g = gf_poly_mul(g, [1, gf_pow(2, i)])
        #print("g ",g)
    return g

def gf_poly_div(dividend, divisor):
    #print("gf_poly_div")
    '''Fast polynomial division by using Extended Synthetic Division and optimized for GF(2^p) computations
    (doesn't work with standard polynomials outside of this galois field, see the Wikipedia article for generic algorithm).'''
    # CAUTION: this function expects polynomials to follow the opposite convention at decoding:
    # the terms must go from the biggest to lowest degree (while most other functions here expect
    # a list from lowest to biggest degree). eg: 1 + 2x + 5x^2 = [5, 2, 1], NOT [1, 2, 5]

    msg_out = list(dividend) # Copy the dividend
    #normalizer = divisor[0] # precomputing for performance
    for i in range(0, len(dividend) - (len(divisor)-1)):
        #msg_out[i] /= normalizer # for general polynomial division (when polynomials are non-monic), the usual way of using
                                  # synthetic division is to divide the divisor g(x) with its leading coefficient, but not needed here.
        coef = msg_out[i] # precaching
        if coef != 0: # log(0) is undefined, so we need to avoid that case explicitly (and it's also a good optimization).
            for j in range(1, len(divisor)): # in synthetic division, we always skip the first coefficient of the divisior,
                                              # because it's only used to normalize the dividend coefficient
                if divisor[j] != 0: # log(0) is undefined
                    msg_out[i + j] ^= gf_mul(divisor[j], coef) # equivalent to the more mathematically correct
                                                               # (but xoring directly is faster): msg_out[i + j] += -divisor[j] * coef

    # The resulting msg_out contains both the quotient and the remainder, the remainder being the size of the divisor
    # (the remainder has necessarily the same degree as the divisor -- not length but degree == length-1 -- since it's
    # what we couldn't divide from the dividend), so we compute the index where this separation is, and return the quotient and remainder.
    separator = -(len(divisor)-1)
    return msg_out[:separator], msg_out[separator:] # return quotient, remainder.


def rs_encode_msg(msg_in, nsym):
    #print("rs_encode_msg1")
    '''Reed-Solomon main encoding function'''
    gen = rs_generator_poly(nsym)

    # Pad the message, then divide it by the irreducible generator polynomial
    _, remainder = gf_poly_div(msg_in + [0] * (len(gen)-1), gen)
    # The remainder is our RS code! Just append it to our original message to get our full codeword (this represents a polynomial of max 256 terms)
    msg_out = msg_in + remainder
    # Return the codeword
    return msg_out


def rs_encode_msg(msg_in, nsym):
    #print("rs_encode_msg2")
    '''Reed-Solomon main encoding function, using polynomial division (algorithm Extended Synthetic Division)'''
    #if (len(msg_in) + nsym) > 255: raise ValueError("Message is too long (%i when max is 255)" % (len(msg_in)+nsym))
    gen = rs_generator_poly(nsym)
    #print("gen ",gen)
    # Init msg_out with the values inside msg_in and pad with len(gen)-1 bytes (which is the number of ecc symbols).
    msg_out = [0] * (len(msg_in) + len(gen)-1)
    # Initializing the Synthetic Division with the dividend (= input message polynomial)
    msg_out[:len(msg_in)] = msg_in

    # Synthetic division main loop
    for i in range(len(msg_in)):
        # Note that it's msg_out here, not msg_in. Thus, we reuse the updated value at each iteration
        # (this is how Synthetic Division works: instead of storing in a temporary register the intermediate values,
        # we directly commit them to the output).
        coef = msg_out[i]

        # log(0) is undefined, so we need to manually check for this case. There's no need to check
        # the divisor here because we know it can't be 0 since we generated it.
        if coef != 0:
            # in synthetic division, we always skip the first coefficient of the divisior, because it's only used to normalize the dividend coefficient (which is here useless since the divisor, the generator polynomial, is always monic)
            for j in range(1, len(gen)):
                #print("gen ",gen[j]," , coef ", coef," , gf ",gf_mul(gen[j], coef))
                msg_out[i+j] ^= gf_mul(gen[j], coef) # equivalent to msg_out[i+j] += gf_mul(gen[j], coef)

    #print("coef ", coef)
    # At this point, the Extended Synthetic Divison is done, msg_out contains the quotient in msg_out[:len(msg_in)]
    # and the remainder in msg_out[len(msg_in):]. Here for RS encoding, we don't need the quotient but only the remainder
    # (which represents the RS code), so we can just overwrite the quotient with the input message, so that we get
    # our complete codeword composed of the message + code.
    msg_out[:len(msg_in)] = msg_in

    return msg_out



def h(k,m):#k and m are vectors
    p=0;
    #print("k ",k, ", len ", len(k), ", m ", m ,", len ", len(m))
    for i in range(0,len(m)-1):
        for j in range(0,len(k)-1):
            #print("i ", i, ", m ",m[i],", j ",j, ", k ",k[j], ", p ",p )
            kj=pow(k[j],j+1)#len(k)-j)
            p=p + int(m[i]*kj)
    return p


# In[7]:


def por(data):
    global portime
    global porst
    
    start=time.time_ns()
    
    r= sympy.randprime(1, 10000)
    random.seed(r)
    g= random.random()
    
    secret_key = os.urandom(len(data))
    encoded_secret_key = base64.b64encode(secret_key)
    
    datascii=[ ord(c) for c in data]    
    #print(datascii)
    
    #print(encoded_secret_key)
    keyascii=[ord(c) for c in str(encoded_secret_key)]
    #print(keyascii)
    
    hmac=h(keyascii,datascii)
    tag= ("r-"+str(r)+"-hamc-"+str(hmac)+"-g-"+str(g))
    msg_in=data+"-tag-"+tag
    #print(msg_in)
    nsym=10#number of errors that can be fixed
    msg_in_ascii=datascii=[ ord(c) for c in msg_in]
    #print(msg_in_ascii)
    
    init_tables()
        
    parity=rs_encode_msg(msg_in_ascii, nsym)
    #print(parity)
    
    end = time.time_ns()
    
    portime=portime+(end - start)
    porst=porst+len(str(parity))-len(str(data))  
    
    #print(portime)
    #print(porst)

por("salam")


# In[8]:


import pandas as pd
import numpy as np

df = pd.read_csv("dat.csv")
ver = pd.read_csv("out.csv")


# In[9]:


bfw=[]
treew=[]
pdpw=[]
porw=[]


# In[10]:


allcols=["","inpatient.number","DestinationDischarge","admission.ward","admission.way","occupation","discharge.department","visit.times","gender","body.temperature","pulse","respiration","systolic.blood.pressure","diastolic.blood.pressure","map","weight","height","BMI","type.of.heart.failure","NYHA.cardiac.function.classification","Killip.grade","myocardial.infarction","congestive.heart.failure","peripheral.vascular.disease","cerebrovascular.disease","dementia","Chronic.obstructive.pulmonary.disease","connective.tissue.disease","peptic.ulcer.disease","diabetes","moderate.to.severe.chronic.kidney.disease","hemiplegia","leukemia","malignant.lymphoma","solid.tumor","liver.disease","AIDS","CCI.score","type.II.respiratory.failure","consciousness","eye.opening","verbal.response","movement","respiratory.support.","oxygen.inhalation","fio2","acute.renal.failure","LVEF","left.ventricular.end.diastolic.diameter.LV","mitral.valve.EMS","mitral.valve.AMS","EA","tricuspid.valve.return.velocity","tricuspid.valve.return.pressure","outcome.during.hospitalization","death.within.28.days","re.admission.within.28.days","death.within.3.months","re.admission.within.3.months","death.within.6.months","re.admission.within.6.months","time.of.death..days.from.admission.","re.admission.time..days.from.admission.","return.to.emergency.department.within.6.months","time.to.emergency.department.within.6.months","creatinine.enzymatic.method","urea","uric.acid","glomerular.filtration.rate","cystatin","white.blood.cell","monocyte.ratio","monocyte.count","red.blood.cell","coefficient.of.variation.of.red.blood.cell.distribution.width","standard.deviation.of.red.blood.cell.distribution.width","mean.corpuscular.volume","hematocrit","lymphocyte.count","mean.hemoglobin.volume","mean.hemoglobin.concentration","mean.platelet.volume","basophil.ratio","basophil.count","eosinophil.ratio","eosinophil.count","hemoglobin","platelet","platelet.distribution.width","platelet.hematocrit","neutrophil.ratio","neutrophil.count","D.dimer","international.normalized.ratio","activated.partial.thromboplastin.time","thrombin.time","prothrombin.activity","prothrombin.time.ratio","fibrinogen","high.sensitivity.troponin","myoglobin","carbon.dioxide.binding.capacity","calcium","potassium","chloride","sodium","Inorganic.Phosphorus","serum.magnesium","creatine.kinase.isoenzyme.to.creatine.kinase","hydroxybutyrate.dehydrogenase.to.lactate.dehydrogenase","hydroxybutyrate.dehydrogenase","glutamic.oxaloacetic.transaminase","creatine.kinase","creatine.kinase.isoenzyme","lactate.dehydrogenase","brain.natriuretic.peptide","high.sensitivity.protein","nucleotidase","fucosidase","albumin","white.globulin.ratio","cholinesterase","glutamyltranspeptidase","glutamic.pyruvic.transaminase","glutamic.oxaliplatin","indirect.bilirubin","alkaline.phosphatase","globulin","direct.bilirubin","total.bilirubin","total.bile.acid","total.protein","erythrocyte.sedimentation.rate","cholesterol","low.density.lipoprotein.cholesterol","triglyceride","high.density.lipoprotein.cholesterol","homocysteine","apolipoprotein.A","apolipoprotein.B","lipoprotein","pH","standard.residual.base","standard.bicarbonate","partial.pressure.of.carbon.dioxide","total.carbon.dioxide","methemoglobin","hematocrit.blood.gas","reduced.hemoglobin","potassium.ion","chloride.ion","sodium.ion","glucose.blood.gas","lactate","measured.residual.base","measured.bicarbonate","carboxyhemoglobin","body.temperature.blood.gas","oxygen.saturation","partial.oxygen.pressure","oxyhemoglobin","anion.gap","free.calcium","total.hemoglobin","GCS","dischargeDay","ageCat"]


# In[11]:


for i in range(0, len(df.index)):
    for j in range(0, len(df.columns)):
        #print(df.iloc[i,j])
        #print(allcols[j])
        tmp=str(i)+","+str(allcols[j])+":"+str(df.iloc[i,j])
        print(tmp)
        if(ver.iloc[i,j]==1): bfw.append(tmp)
        elif(ver.iloc[i,j]==2): treew.append(tmp)
        elif(ver.iloc[i,j]==3): pdpw.append(tmp)
        elif(ver.iloc[i,j]==4): porw.append(tmp)


# In[12]:





# In[13]:


start=time.time_ns()
for d in bfw:
    bloomfilter(d)  
    
end=time.time_ns()
print(end-start)
print(bfst)


# In[14]:


start=time.time_ns()
for d in treew:
    tree(d)  
    
end=time.time_ns()
print(end-start)
print(treest)


# In[ ]:





# In[15]:


start=time.time_ns()
for d in pdpw:
    dpdp(d)  
    
end=time.time_ns()
print(end-start)
print(dpdpst)


# In[16]:


start=time.time_ns()
for d in porw:
    por(d)  
    
end=time.time_ns()
print(end-start)
print(porst)


# gathered data:
# bf:
# 701884000
# 10700
# tree:
# 2252710300
# 493148
# pdp:
# 25767892300
# 3197816
# por
# 15629026600
# 2099105

# In[ ]:


wholetime=701884000+2252710300+25767892300+15629026600
wholestorage=10700+493148+3197816+2099105

print(wholetime)
print(wholestorage)


# In[18]:


wholetime/1000000


# In[20]:


len(bfw)


# In[21]:


len(treew)


# In[ ]:


2099105/3231*


# In[23]:


len(pdpw)


# In[22]:


len(porw)


# In[27]:


pdpall=[]
porall=[]

for i in range(0, len(df.index)):
    for j in range(0, len(df.columns)):
        tmp=str(i)+","+str(allcols[j])+":"+str(df.iloc[i,j])
        pdpall.append(tmp)
        porall.append(tmp)


# In[28]:


start=time.time_ns()
dpdpst=0
for d in pdpall:
    dpdp(d)  
    
end=time.time_ns()
print(end-start)
print(dpdpst)


# In[29]:


start=time.time_ns()
porst=0
for d in porall:
    por(d)  
    
end=time.time_ns()
print(end-start)
print(porst)


# In[ ]:





# In[ ]:





# In[ ]:


#https://gist.github.com/syedrakib/d71c463fc61852b8d366
from Crypto.Cipher import AES
import base64, os

def generate_secret_key_for_AES_cipher():
	# AES key length must be either 16, 24, or 32 bytes long
	AES_key_length = 16 # use larger value in production
	# generate a random secret key with the decided key length
	# this secret key will be used to create AES cipher for encryption/decryption
	secret_key = os.urandom(AES_key_length)
	# encode this secret key for storing safely in database
	encoded_secret_key = base64.b64encode(secret_key)
	return encoded_secret_key

def encrypt_message(private_msg, encoded_secret_key, padding_character):
	# decode the encoded secret key
	secret_key = base64.b64decode(encoded_secret_key)
	# use the decoded secret key to create a AES cipher
	cipher = AES.new(secret_key)
	# pad the private_msg
	# because AES encryption requires the length of the msg to be a multiple of 16
	padded_private_msg = private_msg + (padding_character * ((16-len(private_msg)) % 16))
	# use the cipher to encrypt the padded message
	encrypted_msg = cipher.encrypt(padded_private_msg)
	# encode the encrypted msg for storing safely in the database
	encoded_encrypted_msg = base64.b64encode(encrypted_msg)
	# return encoded encrypted message
	return encoded_encrypted_msg

def decrypt_message(encoded_encrypted_msg, encoded_secret_key, padding_character):
	# decode the encoded encrypted message and encoded secret key
	secret_key = base64.b64decode(encoded_secret_key)
	encrypted_msg = base64.b64decode(encoded_encrypted_msg)
	# use the decoded secret key to create a AES cipher
	cipher = AES.new(secret_key)
	# use the cipher to decrypt the encrypted message
	decrypted_msg = cipher.decrypt(encrypted_msg)
	# unpad the encrypted message
	unpadded_private_msg = decrypted_msg.rstrip(padding_character)
	# return a decrypted original private message
	return unpadded_private_msg


####### BEGIN HERE #######


private_msg = """
 Lorem ipsum dolor sit amet, malis recteque posidonium ea sit, te vis meliore verterem. Duis movet comprehensam eam ex, te mea possim luptatum gloriatur. Modus summo epicuri eu nec. Ex placerat complectitur eos.
"""
padding_character = "{"

secret_key = generate_secret_key_for_AES_cipher()
encrypted_msg = encrypt_message(private_msg, secret_key, padding_character)
decrypted_msg = decrypt_message(encrypted_msg, secret_key, padding_character)

print "   Secret Key: %s - (%d)" % (secret_key, len(secret_key))
print "Encrypted Msg: %s - (%d)" % (encrypted_msg, len(encrypted_msg))
print "Decrypted Msg: %s - (%d)" % (decrypted_msg, len(decrypted_msg))


# In[ ]:



import sympy

# Output : True
print(sympy.isprime(5))                        

# Output : [2, 3, 5, 7, 11, 13, 17, 19, 23, 
# 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 
# 73, 79, 83, 89, 97]
print(list(sympy.primerange(0, 100)))      
    
print(sympy.randprime(0, 100))  # Output : 83
print(sympy.randprime(0, 100)) # Output : 41
print(sympy.prime(3))          # Output : 5
print(sympy.prevprime(50))     # Output : 47
print(sympy.nextprime(50))      # Output : 53

# Output : [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
# 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 
# 79, 83, 89, 97]
print list(sympy.sieve.primerange(0, 100)) 


# In[25]:


oldtime=time.time_ns() / (10 ** 9)
print(oldtime)
for i in range(1,1000):
    newtime=time.time_ns() / (10 ** 9)
    if(oldtime==newtime):
        print(i, end =" ")
    else:
        oldtime=newtime
        print("\n **",i," ** ",newtime," ** ")

