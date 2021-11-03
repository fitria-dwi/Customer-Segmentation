#!/usr/bin/env python
# coding: utf-8

# **Author: Fitria Dwi Wulandari (wulan391@sci.ui.ac.id) - November 01, 2021**

# # Customer Segmentation

# In[3]:


get_ipython().system('pip install kmodes')


# In[4]:


# Import libraries
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.preprocessing import LabelEncoder  
  
from kmodes.kmodes import KModes  
from kmodes.kprototypes import KPrototypes  
  
import pickle  
from pathlib import Path  


# In[8]:


# Import dataset  
customer = pd.read_csv("customer_segments.txt", sep="\t")
print('Data terdiri dari %d kolom dan %d baris.\n' % customer.shape)
customer.head()


# - `Customer_ID`: Kode pelanggan.
# - `Nama Pelanggan`: Nama dari pelanggan.
# - `Jenis Kelamin`: Jenis kelamin dari pelanggan, bertipe kategori yang terdiri dari Pria dan Wanita.
# - `Umur`: Umur dari pelanggan.
# - `Profesi`: Profesi dari pelanggan, bertipe kategori yang terdiri dari Wiraswasta, Pelajar, Professional, Ibu Rumah Tangga, dan Mahasiswa.
# - `Tipe Residen`: Tipe tempat tinggal dari pelanggan kita, bertipe kategori yang terdiri dari Cluster dan Sector.
# - `Nilai Belanja Setahun`:  Total belanja yang sudah dikeluarkan oleh pelanggan tersebut.

# In[7]:


# Menampilkan informasi data  
customer.info()


# - Tidak ada nilai null pada data.
# - Dua kolom memiliki tipe data numerik dan lima data bertipe string.

# ### Eksplorasi Data

# #### Eksplorasi Data Numerik

# In[29]:


sns.set(style='white')
plt.clf()
  
def observasi_num(features):  
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    for i, kol in enumerate(features):
        sns.boxplot(customer[kol], ax = axs[i][0])
        sns.distplot(customer[kol], ax = axs[i][1])   
        axs[i][0].set_title('mean = %.2f\n median = %.2f\n std = %.2f'%(customer[kol].mean(), customer[kol].median(), customer[kol].std()))
    plt.setp(axs)
    plt.tight_layout()
    plt.show()  

kolom_numerik = ['Umur','NilaiBelanjaSetahun'] 
observasi_num(kolom_numerik) 


# #### Eksplorasi Data Kategorik

# In[30]:


sns.set(style='white')
plt.clf()
 
kolom_kategorikal = ['Jenis Kelamin','Profesi', 'Tipe Residen']  

fig, axs = plt.subplots(3,1,figsize=(7,10)) 

# Membuat plot untuk setiap kolom kategorikal  
for i, kol in enumerate(kolom_kategorikal):  
    sns.countplot(customer[kol], order = customer[kol].value_counts().index, ax = axs[i])
    axs[i].set_title('\nCount Plot %s\n'%(kol), fontsize=15)    
       
    for p in axs[i].patches:  # Memberikan anotasi  
        axs[i].annotate(format(p.get_height(), '.0f'),  
                        (p.get_x() + p.get_width() / 2., p.get_height()),  
                        ha = 'center',  
                        va = 'center',  
                        xytext = (0, 10),  
                        textcoords = 'offset points') 
           
    sns.despine(right=True,top = True, left = True)  # Setting plot 
    axs[i].axes.yaxis.set_visible(False) 
    plt.setp(axs)
    plt.tight_layout()

plt.show()


# **Summary:**
# - Rata-rata dari umur pelanggan adalah 37.5 tahun.
# - Rata-rata dari nilai belanja setahun pelanggan adalah 7,069,874.82.
# - Jenis kelamin pelanggan di dominasi oleh wanita sebanyak 41 orang (82%) dan laki-laki sebanyak 9 orang (18%).
# - Profesi pelanggan di dominasi oleh Wiraswasta (40%) diikuti dengan Professional (36%) dan lainnya sebanyak (24%).
# - Dari seluruh pelanggan 64% dari mereka tinggal di Cluster dan 36% nya tinggal di Sektor.

# ### Menyiapkan Data untuk Permodelan

# #### Standardisasi Kolom Numerik

# **Note:** Standardisasi perlu dilakukan apabila menggunakan permodelan dengan teknik *unsupervised clustering*. Standardisasi dilakukan agar data yang digunakan memiliki skala yang sama antar variabelnya sehingga variabel yang memiliki skala besar tidak mendominasi bagaimana cluster akan dibentuk dan juga tiap variabel akan dianggap sama pentingnya oleh algoritma yang akan digunakan.

# In[36]:


from sklearn.preprocessing import StandardScaler  
  
kolom_numerik = ['Umur', 'NilaiBelanjaSetahun']  
  
# Sebelum standardisasi  
print('Informasi sebelum Standardisasi\n')  
print(customer[kolom_numerik].describe().round(1))  
  
# Standardisasi  
customer_std = StandardScaler().fit_transform(customer[kolom_numerik])  

customer_std = pd.DataFrame(data=customer_std, index=customer.index, columns=customer[kolom_numerik].columns)  

print('\nContoh hasil standardisasi\n')  
print(customer_std.head())  
print('\nInformasi setelah standardisasi\n')  
print(customer_std.describe().round(0))  


# #### Konversi Data Kategorik dengan Label Encoder

# **Note:**  *Encoding* kolom-kolom kategorik menjadi numerik juga diperlukan dalam menggunakan teknik *unsupervised clustering*.

# In[37]:


from sklearn.preprocessing import LabelEncoder
  
kolom_kategorik = ['Jenis Kelamin','Profesi','Tipe Residen']  
  
# Salinan data frame  
customer_encode = customer[kolom_kategorikal].copy()  
  
for col in kolom_kategorikal:  
    customer_encode[col] = LabelEncoder().fit_transform(customer_encode[col])
       
print(customer_encode.head())


# #### Menggabungkan Data untuk Permodelan

# In[38]:


# Menggabungkan data frame
customer_model = customer_encode.merge(customer_std, left_index = True, right_index=True, how = 'left')  
print (customer_model.head())


# ### Membangun Model dengan Algoritma Kprototype

# #### Mencari Jumlah Cluster yang Optimal

# In[40]:


# Melakukan iterasi untuk mendapatkan nilai Cost  
cost = {}
for k in range(2,10):
  kproto = KPrototypes(n_clusters = k,random_state=75)
  kproto.fit_predict(customer_model, categorical=[0,1,2])
  cost[k]= kproto.cost_ 
  
# Visualisasi elbow plot  
sns.pointplot(x=list(cost.keys()), y=list(cost.values()))  
plt.show()


# Terlihat bahwa garis mengalami patahan yang membentuk elbow atau siku pada saat k = 5. Maka dengan menggunakan metode ini diperoleh k optimal pada saat berada di k = 5.

# #### Membangun Model

# In[41]:


kproto = KPrototypes(n_clusters=5, random_state = 75)
kproto = kproto.fit(customer_model, categorical=[0,1,2])

#Save Model
pickle.dump(kproto, open('cluster.pkl', 'wb'))


# #### Mengimplementasikan Model

# Model yang sudah dibuat digunakan untuk menentukan segmen atau cluster pelanggan yang ada di dataset.

# In[42]:


# Menentukan segmen tiap pelanggan    
clusters = kproto.predict(customer_model, categorical=[0,1,2])
print('segmen pelanggan: {}\n'.format(clusters))

# Menggabungkan data awal dan segmen pelanggan
customer_final = customer.copy()
customer_final['cluster'] = clusters
print(customer_final.head()) 


# In[43]:


# Menampilkan data pelanggan berdasarkan cluster nya  
for i in range (0,5):
    print('\nPelanggan cluster: {}\n'.format(i))
    print(customer_final[customer_final['cluster']== i])


# #### Visualisasi Hasil Clustering - Box Plot

# In[44]:


# Data Numerik
kolom_numerik = ['Umur','NilaiBelanjaSetahun']  

for i in kolom_numerik:
    plt.figure(figsize=(6,4))
    ax = sns.boxplot(x = 'cluster',y = i, data = customer_final)
    plt.title('\nBox Plot {}\n'.format(i), fontsize=12)
    plt.show()


# #### Visualisasi Hasil Clustering - Count Plot

# In[45]:


# Data Kategorik 
kolom_categorical = ['Jenis Kelamin','Profesi','Tipe Residen']  
  
for i in kolom_categorical:
    plt.figure(figsize=(6,4))
    ax = sns.countplot(data = customer_final, x = 'cluster', hue = i )
    plt.title('\nCount Plot {}\n'.format(i), fontsize=12)
    ax.legend(loc="upper center")
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
    (p.get_x() + p.get_width() / 2., p.get_height()),
    ha = 'center',
    va = 'center',
    xytext = (0, 10),
    textcoords = 'offset points')

sns.despine(right=True,top = True, left = True)
ax.axes.yaxis.set_visible(False)
plt.show()


# #### Pemetaan Cluster

# In[48]:


# Mapping
customer_final['segmen'] = customer_final['cluster'].map({
0: 'Silver Students',
1: 'Diamond Senior Entrepreneur',
2: 'Gold Senior Member',
3: 'Diamond Young Entrepreneur',
4: 'Gold Young Member'
})

print(customer_final.info())
print(customer_final.head())


# **Summary:**
# - **Cluster 0**: *Silver Students*, cluster ini terdiri dari para pelajar dan mahasiswa dengan rata-rata umur mereka adalah 16 tahun dan nilai belanja setahun mendekati 3 juta.
# - **Cluster 1**: *Diamond Senior Entrepreneur*, cluster ini terdiri dari para wiraswasta yang memiliki nilai transaksi rata-rata mendekati 10 juta dengan rentang umur sekitar 45 - 64 tahun dan rata-ratanya adalah 55 tahun.
# - **Cluster 2**: *Gold Senior Member*, cluster ini terdiri dari para profesional dan ibu rumah tangga yang berusia tua dengan rentang umur 46 - 63 tahun dan dengan rata-rata 53 tahun dan nilai belanja setahunnya mendekati 6 juta.
# - **Cluster 3**: *Diamond Young Entrepreneur*, cluster ini terdiri dari para wiraswasta yang memiliki nilai transaksi rata-rata mendekati 10 juta dengan rentang umur sekitar 18 - 41 tahun dan rata-ratanya adalah 29 tahun.
# - **Cluster 4**: *Gold Young Member*, cluster ini terdiri dari para profesional dan ibu rumah tangga yang berusia muda dengan rentang umur sekitar 20 - 40 tahun dan dengan rata-rata 30 tahun dan nilai belanja setahunnya mendekati 6 juta.
