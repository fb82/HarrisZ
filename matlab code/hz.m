function kpt_vl=hz(im)

scale_base=sqrt(2);
scale_ratio=1/sqrt(2);
scale_th=0.75;
n_scales=9;
start_scale=3;
dm_th=0.31;
cf=3;

kpt=[];
img=single(rgb2gray(im));
i_scale=scale_base.^(0:n_scales);
d_scale=i_scale*scale_ratio;
[dx,dy]=derivative(img);
for i=start_scale+1:n_scales
    r_d=max(1,ceil(3*d_scale(i)));
    hd=2*r_d+1;
    dx_d=imgaussfilt(dx,d_scale(i),'FilterSize',[hd hd],'Padding','symmetric');
    dy_d=imgaussfilt(dy,d_scale(i),'FilterSize',[hd hd],'Padding','symmetric');
    dm=(dx_d.^2+dy_d.^2).^0.5;
    dm_mask=imgaussfilt(single(dm>mean(dm(:))),d_scale(i),'FilterSize',[hd hd],'Padding','symmetric');   
    dx_d=dx_d.*dm_mask;
    dy_d=dy_d.*dm_mask;
    hi=2*max(1,ceil(3*i_scale(i)))+1;
    dxy=imgaussfilt(dx_d.*dy_d,i_scale(i),'FilterSize',[hi hi],'Padding','symmetric');
    dx2=imgaussfilt(dx_d.^2,i_scale(i),'FilterSize',[hi hi],'Padding','symmetric');
    dy2=imgaussfilt(dy_d.^2,i_scale(i),'FilterSize',[hi hi],'Padding','symmetric');
    harris=zscore(dx2.*dy2-dxy.^2)-zscore((dx2+dy2).^2);
    kp=max_mask(harris,r_d,dm_mask>dm_th);
    kp_sub_pix=sub_pix(harris,kp);
    kp_s=repmat([i-1 d_scale(i) i_scale(i)],size(kp,1),1);
    kp_idx=sub2ind(size(dxy),kp(:,1),kp(:,2));
    [kp_eli,kp_ratio]=get_eli(dx2(kp_idx),dy2(kp_idx),dxy(kp_idx),i_scale(i)*cf);
    kpt_=[kp_sub_pix(:,[2 1]) kp_eli kp_s kp_ratio];
    kp_good=1-kp_ratio<scale_th;
    kpt=[kpt; kpt_(kp_good,:)];
end

for i=1:size(kpt,1)
    aux=inv([kpt(i,3) kpt(i,4); kpt(i,4) kpt(i,5)]);
    kpt(i,[3 4 5])=aux([1 2 4]);
end
kpt_vl=kpt(:,[1 2 3 4 5]);
kpt_vl=kpt_vl';

function [kp_eli,kp_ratio]=get_eli(dx2,dy2,dxy,scale)

kp_eli=zeros(size(dx2,1),3);
kp_ratio=zeros(size(dx2,1),1);

for i=1:size(dx2,1)
    U=[dx2(i) dxy(i); dxy(i) dy2(i)];
    [V,D]=eig(U);
    D1=1/sqrt(D(1,1));
    D2=1/sqrt(D(2,2));
    D_max=max(D1,D2);
    D1=D1/D_max;
    D2=D2/D_max;
    kp_ratio(i)=min(D1,D2);
    D1=D1*scale;
    D2=D2*scale;
    D=[1/D1^2 0; 0 1/D2^2];
    U_=V*D*V';
    kp_eli(i,:)=[U_(1,1) U_(1,2) U_(2,2)];
end
    
function kp_sp=sub_pix(img,kp)

r_mask=(kp(:,1)>1)&(kp(:,1)<size(img,1));
c_mask=(kp(:,2)>1)&(kp(:,2)<size(img,2));
kp_sp=kp;

v=img(sub2ind(size(img),kp(r_mask,1),kp(r_mask,2)));
vl=img(sub2ind(size(img),kp(r_mask,1)-1,kp(r_mask,2)));
vr=img(sub2ind(size(img),kp(r_mask,1)+1,kp(r_mask,2)));
kp_sp(r_mask,1)=kp(r_mask,1)+(-vl+vr)./(2*(-vl-vr+2*v));

v=img(sub2ind(size(img),kp(c_mask,1),kp(c_mask,2)));
vl=img(sub2ind(size(img),kp(c_mask,1),kp(c_mask,2)-1));
vr=img(sub2ind(size(img),kp(c_mask,1),kp(c_mask,2)+1));
kp_sp(c_mask,2)=kp(c_mask,2)+(-vl+vr)./(2*(-vl-vr+2*v));

function rc=max_mask(img,r_d,dm_mask)

rad_max=3;
rad=min(rad_max,max(1,round(r_d/sqrt(2))));
k=ones(2*rad+1,2*rad+1,'logical');
k(rad+1,rad+1,1)=0;
ismax=(img>imdilate(img,k,'same'))&(img>0)&dm_mask;
[r,c]=find(ismax);
[~,idx]=sort(img(ismax),'descend');
rc=[r(idx) c(idx)];
m=pdist2(rc,rc);
m_idx=zeros(size(idx,1),1);
m_idx(1)=1;
m_n=1;
for i=2:size(m_idx,1)
    if all(m(i,m_idx(1:m_n))>=r_d)
        m_n=m_n+1;
        m_idx(m_n)=i;
    end
end
rc(m_idx(1:m_n),:);

function img=zscore(img)

s=std(img(:));
if s
    img=(img-mean(img(:)))/s;
else
    img(:)=0;
end

function [dx,dy]=derivative(img)

kx=[...
    0 0  0;...
    1 0 -1;...
    0 0  0;...
    ];

ky=[...
    0  1 0;...
    0  0 0;...
    0 -1 0;...
    ];

dx=conv2(img,kx,'same');
dy=conv2(img,ky,'same');

dx(1:end,1)=0;
dx(1:end,end)=0;
dx(1,1:end)=0;
dx(end,1:end)=0;

dy(1:end,1)=0;
dy(1:end,end)=0;
dy(1,1:end)=0;
dy(end,1:end)=0;