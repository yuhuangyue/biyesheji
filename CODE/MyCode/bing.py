#coding=utf-8
#必应图片爬虫
import re
import os
import urllib.request
import requests

coding = 'utf-8'
thepath='C:/Users/56891/Desktop/BBCnews/BBCnews_20130721/temp/'

def getPages(current):

    urlopenheader = {'User-Agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:42.0) Gecko/20100101 Firefox/42.0'}
    url = 'http://cn.bing.com/images/async?q=奥运会100米短跑' + '&async=content&first=' + str(current)
    response = requests.get(url, None, headers=urlopenheader)
    html = response.text
    #print(html)
    print ('\n')
    return html

# 获得图片地址
def getImg(html,current):
       # reg = r'src="(.*?\.jpg)" size="'        # 定义一个正则来匹配页面当中的图片
       # reg = '(&quot;murl&quot;:&quot;http://.*?.(jpg|png|jpeg)(&quot|/0&quot);)'
        reg = '(&quot;(.*?)&quot;)'
        imgre = re.compile(reg)         # 为了让正则更快，给它来个编译
        #这个时候做个测试，把匹配的数据都给打印出来
        imglist = re.findall(imgre, html)                       # 通过正则返回所有数据列表
        

        reg2 = '(http://.*?.(jpg|png|jpeg))'
        imgre2 = re.compile(reg2)
        

        x = 0 
        for imgurl in imglist:
              #  urllib.urlretrieve(imgurl,'%s.jpg' % x)
                imgurl = re.findall(imgre2,imgurl[0])
                #print (imgurl)

                try:
                    pic= urllib.request.urlopen(imgurl[0][0],timeout=10).read()
                except:
                    continue
                file = thepath +str(current)+ str(x) + '.jpg'
                fp = open(file,'wb')
                fp.write(pic)
                fp.close()
                
                
                #print (imgurl[0][0])
                print ('\n')
                x+=1

for i in range(1,5):
    html = getPages(i+30)
    getImg(html,i)
