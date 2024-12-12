function kpt_vl=hz_plus(im,max_kpts,fast_save_memory)
% kpt_vl = [ x y eli_a eli_b eli_c ]'

% max_kpts=8000;      % needed to set the uniform distribution  
% fast_save_memory=0; % unless big image with a lot of kpts

% x,y coordinates are according matlab convention, i.e. the coordinates of
% the first pixel in the image are [1,1]
xy_offset=-0.5;

% old parameters
start_scale=0;
n_scales=4;
scale_base=sqrt(2);
scale_ratio=1/sqrt(2);
scale_th=0.75;
dm_th=0.31;
cf=3;

% new parameters
max_kpts_cf=2;
start_scale_at_2x=2;
rescale_method='lanczos3';
min_scale=sqrt(2);
sieve_rad=1;
laf_offset=10;

% gradient mask for color images
color_grad=1;
if numel(size(im))~=3
    color_grad=0;
end

sz=size(im);
    
if start_scale<start_scale_at_2x
    im_2x=imresize(im,2,rescale_method);
end    

kpt=[];
if color_grad
    img1=single(rgb2gray(im));
    img2=single(max(im,[],3));
    [dx1,dy1]=derivative(img1);
    [dx2,dy2]=derivative(img2);
    [dx_g,dy_g]=best_derivative(dx1,dy1,dx2,dy2);    
    dx=dx1;
    dy=dy1;
    
    if start_scale<start_scale_at_2x
        img1_2x=single(rgb2gray(im_2x));
        img2_2x=single(max(im_2x,[],3));
        [dx1_2x,dy1_2x]=derivative(img1_2x);
        [dx2_2x,dy2_2x]=derivative(img2_2x);
        [dx_g_2x,dy_g_2x]=best_derivative(dx1_2x,dy1_2x,dx2_2x,dy2_2x);    
        dx_2x=dx1_2x;
        dy_2x=dy1_2x;        
    end
else
    if numel(size(im))==3
		img=single(rgb2gray(im));
    else
		img=im;
	end
    [dx,dy]=derivative(img);     

    if start_scale<start_scale_at_2x
        if numel(size(im_2x))==3
            img_2x=single(rgb2gray(im_2x));
        else
            img_2x=im_2x;
        end
        [dx_2x,dy_2x]=derivative(img_2x);     
    end
end
i_scale=scale_base.^(start_scale:n_scales);
d_scale=i_scale*scale_ratio;
is_2x_scale=(start_scale:n_scales)<start_scale_at_2x;

dx_1x=dx;
dy_1x=dy;
if color_grad
    dx_g_1x=dx_g;
    dy_g_1x=dy_g;
end

for i=1:length(i_scale)
    if ~is_2x_scale(i)
        dx=dx_1x;
        dy=dy_1x;
        if color_grad
            dx_g=dx_g_1x;
            dy_g=dy_g_1x;
        end        
    else
        dx=dx_2x;
        dy=dy_2x;
        if color_grad
            dx_g=dx_g_2x;
            dy_g=dy_g_2x;
        end   
        d_scale(i)=d_scale(i)*2;
        i_scale(i)=i_scale(i)*2;
    end   
    r_d=max(1,ceil(3*d_scale(i)));
    hd=2*r_d+1;
    dx_d=imgaussfilt(dx,d_scale(i),'FilterSize',[hd hd],'Padding','symmetric');
    dy_d=imgaussfilt(dy,d_scale(i),'FilterSize',[hd hd],'Padding','symmetric');
    if ~color_grad
        dm=(dx_d.^2+dy_d.^2).^0.5;
        dm=dm>mean(dm(:));
        dm_mask=imgaussfilt(single(dm),d_scale(i),'FilterSize',[hd hd],'Padding','symmetric');   
    else
        dx_dg=imgaussfilt(dx_g,d_scale(i),'FilterSize',[hd hd],'Padding','symmetric');
        dy_dg=imgaussfilt(dy_g,d_scale(i),'FilterSize',[hd hd],'Padding','symmetric');
        dm=(dx_dg.^2+dy_dg.^2).^0.5;
        dm=dm>mean(dm(:));
        dm_mask=imgaussfilt(single(dm),d_scale(i),'FilterSize',[hd hd],'Padding','symmetric');   
    end    
    dx_d=dx_d.*dm_mask;
    dy_d=dy_d.*dm_mask;
    hi=2*max(1,ceil(3*i_scale(i)))+1;
    dxy=imgaussfilt(dx_d.*dy_d,i_scale(i),'FilterSize',[hi hi],'Padding','symmetric');
    dx2=imgaussfilt(dx_d.^2,i_scale(i),'FilterSize',[hi hi],'Padding','symmetric');
    dy2=imgaussfilt(dy_d.^2,i_scale(i),'FilterSize',[hi hi],'Padding','symmetric');
    harris=zscore(dx2.*dy2-dxy.^2)-zscore((dx2+dy2).^2);
    dm_filt_mask=dm_mask>dm_th;
    if ~fast_save_memory
        kp=max_mask(harris,r_d,dm_filt_mask);
    else
        kp=max_mask_fast(harris,r_d,dm_filt_mask);
    end
    hv=harris(sub2ind(size(harris),kp(:,1),kp(:,2)));         
    kp_sub_pix=sub_pix(harris,kp);
    
    if ~is_2x_scale(i)    
        kp_s=repmat([i max(min_scale*scale_ratio,d_scale(i)) max(min_scale,i_scale(i))],size(kp,1),1);
        kp_idx=sub2ind(size(dxy),kp(:,1),kp(:,2));
        [kp_eli,kp_ratio]=get_eli(dx2(kp_idx),dy2(kp_idx),dxy(kp_idx),max(min_scale,i_scale(i))*cf);
        kpt_=[kp_sub_pix(:,[2 1]) kp_eli kp_s kp_ratio hv];
        kp_good=1-kp_ratio<scale_th;
        kpt=[kpt; kpt_(kp_good,:)];
    else
        kp_s=repmat([i max(min_scale*scale_ratio,d_scale(i)/2) max(min_scale,i_scale(i)/2)],size(kp,1),1);
        kp_idx=sub2ind(size(dxy),kp(:,1),kp(:,2));
        [kp_eli,kp_ratio]=get_eli(dx2(kp_idx),dy2(kp_idx),dxy(kp_idx),max(min_scale,i_scale(i)/2)*cf);
        kpt_=[kp_sub_pix(:,[2 1])/2 kp_eli kp_s kp_ratio hv];
        kp_good=1-kp_ratio<scale_th;
        kpt=[kpt; kpt_(kp_good,:)];        
    end
end

if min_scale
    kpt=sortrows(kpt,10,'descend');
    to_check=kpt(:,7)==min_scale*scale_ratio;
    rc=kpt(to_check,:);
    if ~fast_save_memory
        rc=greedy_max(rc,sieve_rad);
    else
        rc=select_max(rc,sieve_rad,size(im));    
    end    
    kpt=[kpt(~to_check,:); rc(:,1:size(kpt,2))]; 
    kpt=sortrows(kpt,10,'descend');
end

kpt(:,[1 2])=kpt(:,[1 2])+xy_offset;

if ~fast_save_memory
    kpt=uniform_kpts(im,kpt,max_kpts,max_kpts_cf);
else
    rp=ceil(kpt(:,8)*cf*3)+laf_offset+1;
    pm_=prod(1-min(1,max(0,[(rp-kpt(:,1))./rp (rp-kpt(:,2))./rp...
        (rp+kpt(:,1)-sz(2))./rp (rp+kpt(:,2)-sz(1))./rp])),2)==1;        
    kpt=uniform_kpts_big(im,[kpt pm_],max_kpts,max_kpts_cf);
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
if ~isempty(rc)
    rc=rc(m_idx(1:m_n),:);
end

function rc=max_mask_fast(img,r_d,dm_mask)

rad_max=3;
rad=min(rad_max,max(1,round(r_d/sqrt(2))));
k=ones(2*rad+1,2*rad+1,'logical');
k(rad+1,rad+1,1)=0;
ismax=(img>imdilate(img,k,'same'))&(img>0)&dm_mask;
[r,c]=find(ismax);
if isempty(r)
    rc=[];
    return;
end
[img_val,idx]=sort(img(ismax),'descend');
rc=single([r(idx) c(idx) img_val]);
rc=select_max(rc,r_d,size(img));

function rc=select_max(rc,r_d,sz)

bblock=2^14;

if size(rc,1)<=bblock
    rc=greedy_max(rc,r_d);
else
    rc_out=[];
    step=1024;
    for ri=1:step:sz(1)
        ril_in=ri;
        rir_in=min(ri+step-1,sz(1));            
        rir_out=rir_in+r_d+1;            
        if (rir_in==sz(1))
            ril_out=sz(1)-step+1-r_d-1;
        else
            ril_out=ril_in-r_d-1;
        end        
        for ci=1:step:sz(2)                
            cil_in=ci;
            cir_in=min(ci+step-1,sz(2));            
            cir_out=cir_in+r_d+1;            
            if (cir_in==sz(2))
                cil_out=sz(2)-step+1-r_d-1;
            else
                cil_out=cil_in-r_d-1;
            end
            
            rc_=rc((rc(:,1)>=ril_out)&(rc(:,1)<=rir_out)&(rc(:,2)>=cil_out)&(rc(:,2)<=cir_out),:);
            if ~isempty(rc_)
                rc_=greedy_max(rc_,r_d);
            end
            if ~isempty(rc_)
                rc_=rc_((rc_(:,1)>=ril_in)&(rc_(:,1)<=rir_in)&(rc_(:,2)>=cil_in)&(rc_(:,2)<=cir_in),:);
            end
            if ~isempty(rc_)
                rc_out=[rc_out; rc_];
            end
        end
    end
    [~,idx]=sort(rc_out(:,3),'descend');
    rc=rc_out(idx,:);
end

function [rc,m_idx]=greedy_max(rc,r_d)

m=rangesearch(rc(:,[1 2]),rc(:,[1 2]),r_d-eps);
m_idx=zeros(size(m,1),1);
m_idx(1)=1;
m_n=1;
for i=2:size(m_idx,1)
    if isempty(intersect(m{i},m_idx(1:m_n)))    
        m_n=m_n+1;
        m_idx(m_n)=i;
    end
end
rc=rc(m_idx(1:m_n),:);
m_idx=m_idx(1:m_n);

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

function [dx,dy]=best_derivative(dx1,dy1,dx2,dy2)

aux=reshape([dx1(:); dx2(:)],[size(dx1) 2]);
dx=max_abs(aux);
aux=reshape([dy1(:); dy2(:)],[size(dy1) 2]);
dy=max_abs(aux);

function r=max_abs(img)

[~,idx]=max(abs(img),[],3);
r=zeros(size(img,1),size(img,2),'single');

for i=1:size(img,3)
    aux=img(:,:,i);
    r(idx==i)=aux(idx==i);
end

function kpt=uniform_kpts(im,kpt,max_kpts,max_kpts_cf)

c_kp=[];
r_kp=kpt;
c_d=2*sqrt(size(im,1)*size(im,2)/(max_kpts*pi/max_kpts_cf));
while 1
    if isempty(r_kp)
        break;
    end        
    [~,idx]=sortrows([r_kp(:,10) r_kp(:,6)],[-1 -2]);
    r_kp=r_kp(idx,:);
    rc=r_kp(:,[1 2]);
    m=pdist2(rc,rc);
    m_idx=zeros(size(idx,1),1);
    m_idx(1)=1;
    m_n=1;
    for i=2:size(m_idx,1)
        if all(m(i,m_idx(1:m_n))>=c_d)
            m_n=m_n+1;
            m_idx(m_n)=i;
        end
    end
    c_idx=zeros(size(idx,1),1,'logical');    
    if ~isempty(rc)
        c_idx(m_idx(1:m_n))=1;
    end    
    c_kp=[c_kp; r_kp(c_idx,:)];
    r_kp(c_idx,:)=[];
end
kpt=c_kp;

function kpt=uniform_kpts_big(im,kpt,max_kpts,max_kpts_cf)

bblock=2^11;

c_kp=[];
r_kp=kpt;
c_d=2*sqrt(size(im,1)*size(im,2)/(max_kpts*pi/max_kpts_cf));
while 1
    if isempty(r_kp)
        break;
    end        
    [~,idx]=sortrows([r_kp(:,10) r_kp(:,6)],[-1 -2]);
    r_kp=r_kp(idx,:);
    rc=r_kp(:,[1 2]);
    if size(rc,1)<bblock    
        m=pdist2(rc,rc);
        m_idx=zeros(size(idx,1),1);
        m_idx(1)=1;
        m_n=1;
        for i=2:size(m_idx,1)
            if all(m(i,m_idx(1:m_n))>=c_d)
                m_n=m_n+1;
                m_idx(m_n)=i;
            end
        end
    else
        m_idx=zeros(size(idx,1),1);
        m_idx(1)=1;
        m_n=1;
        max_offset=bblock;
        m_offset=[1 max_offset];        
        m=pdist2(single(rc(m_offset(1):m_offset(2),:)),single(rc));
        for i=2:size(m_idx,1)
            if i>m_offset(2)
                m_offset=m_offset+max_offset;
                m=pdist2(single(rc(m_offset(1):min(size(m_idx,1),m_offset(2)),:)),single(rc));
            end
            if all(m(i-m_offset(1)+1,m_idx(1:m_n))>=c_d)
                m_n=m_n+1;
                m_idx(m_n)=i;
            end
        end        
    end    
    c_idx=zeros(size(idx,1),1,'logical');
    if ~isempty(rc)
        c_idx(m_idx(1:m_n))=1;
    end    
    c_kp=[c_kp; r_kp(c_idx,:)];
    r_kp(c_idx,:)=[];
    if sum(c_kp(:,11))>max_kpts*1.2
        break;
    end
end
kpt=c_kp;
