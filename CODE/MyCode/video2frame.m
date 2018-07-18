%  clc;
%  clear;
% %
% load Summe
%
% for pos = 1:19
%     video_name = filelist{pos};
% %% 读取视频
%     video_file= strcat('C:\Users\56891\Desktop\BBCnews\',video_name);
%     video=VideoReader(video_file);
%     frame_number=floor(video.Duration * video.FrameRate);
% %% 分离图片
%     temp = strsplit(video_name,'.')
%     video_name = temp(1)
%     video_name = video_name{1}
%     ff =  strcat('C:\Users\56891\Desktop\BBCnews\',video_name)
%     if ~exist(ff)
%         mkdir (ff)         % 若不存在，在当前目录中产生一个子目录‘Figure’
%
%
%     for i=1:frame_number
%         image_name=strcat('C:\Users\56891\Desktop\BBCnews\',video_name,'\',num2str(i));
%         image_name=strcat(image_name,'.jpg');
%         I=read(video,i);                               %读出图片
%         imwrite(I,image_name,'jpg');                   %写图片
%         I=[];
%
%     end
%     end
% end



video=VideoReader('C:\KK_Movies\kk 2018-06-08 18-50-02.avi');
frame_number=floor(video.Duration * video.FrameRate);
%% 分离图片
index= [] ;
pos = 1;
for i=1:24:frame_number
    image_name=strcat('C:\KK_Movies\',num2str(i));
    image_name=strcat(image_name,'.jpg');
    I=read(video,i);                               %读出图片
    imwrite(I,image_name,'jpg');                   %写图片
    I=[];
    index (pos,1) = i;
    pos = pos +1;

end

%  下面这段代码计算正确率的
% acc = 0;
%
% for i = 1:152
%     if index(i,1)>index(i,2) && index(i,3)==1
%         acc = acc +1;
%     end
%     if index(i,1)<index(i,2) && index(i,3)==0
%         acc = acc +1;
%     end
%
% end
%
% acc/152


% 下面这代码计算mAP
% pos = 1;
% AP = 0;
% acc = 0
% for i = 1:220
%         if index (i,1)>index(i,2) && index (i,4)==1
%             AP = pos/index(i,3)+AP
%             pos = pos +1
%             acc = acc +1
%         end
% 
%     
% end
% acc/220
% AP/pos

% %下面这段代码计算F-score
% a = 0;
% b = 0;
% a2 = 0;
% b2 = 0;
% for i=1:151
%     if index(i,1)>index(i,2) && index(i,4)~=0
%         a2=a2+1;
%     end
%     
%     if index(i,1)<index(i,2) && index(i,4)==0
%         b2=b2+1;
%     end
%     
%     if index(i,4)~=0
%         b=b+1;
%     else
%         a = a+1;
%     end
%     
% end
% 
% precious = a2/(a2+b2);
% recall = a2/a;
% fscore=(2*precious*recall)/(precious+recall)

