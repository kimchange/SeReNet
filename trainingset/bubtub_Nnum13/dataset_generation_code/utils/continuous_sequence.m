function DrawingPoint=continuous_sequence(Magnitude, seqlength , times_interp)
    arguments
        Magnitude (1,1) double = 10
        seqlength (1,1) double = 200
        times_interp (1,1) double = 10
    end
% InputDots = [1,1+j,-1+j,-1]; % 这里是点
% 
% i=-pi:0.3:pi;
% x=2.*(sin(i)-sin(2*i)./2);
% y=2.*(cos(i)-cos(i).^2);
InputDots = Magnitude*randn(1,seqlength / times_interp);

% figure;hold on;axis([-4,4,-4,4])
% for i = 1:length(InputDots)
%     plot(i,InputDots(i),'*');pause(0.05);
% end

%InputDots = [1,j,-1,0]; % 这里是点
% times_interp = 100; % 修改这里，越大越平滑
numDrawingPoint = times_interp*length(InputDots);
RotationFactor = ones(1,length(InputDots));

% for i = 1:length(InputDots)
%     RotationFactor(i) = exp(j*2*pi*(i-1)/length(InputDots) / times_interp);
%     
% end
i = 1:length(InputDots);
i = i- ceil( (length(InputDots) + 1) /2);
i = ifftshift(i);
RotationFactor = exp(j*2*pi*(i)/length(InputDots) / times_interp);

% RotationFactor = [1,exp(j*2*pi/4),exp(j*2*pi/4*2),exp(j*2*pi/4*3)];
% Vector_0 即所有4个向量在0时刻的状态，包括4个向量的幅度和相位
% Vector_0 = fft(InputDots,numDrawingPoint)/numDrawingPoint;
Vector_0 = fft(InputDots)/length(InputDots);
% % 与旋转因子相乘，代表4个向量不同的转速，逆时针为正，分别为1个周期转0圈，1圈，2圈，3圈
% % （fourier级数绘图更好得展示了旋转因子的物理意义）
% Vector_1 = Vector_0.*RotationFactor;
% Vector_2 = Vector_1.*RotationFactor;
% Vector_3 = Vector_2.*RotationFactor;
% Vector_4 = Vector_3.*RotationFactor;
% % sum相当于在复数平面把 Vector_n 中的所有复数向量首尾相连


DrawingVector = zeros(numDrawingPoint,length(InputDots));
DrawingVector(1,:) = Vector_0.*RotationFactor;
for i = 2:numDrawingPoint
    DrawingVector(i,:) = DrawingVector(i-1,:).*RotationFactor;
    
end
DrawingPoint = [sum(Vector_0) sum(DrawingVector,2).'];
DrawingPoint = real(DrawingPoint(1:end-1));
DrawingPoint = DrawingPoint / max(abs(DrawingPoint(:))) * Magnitude;
% figure
% plot(InputDots)
% figure
% plot(DrawingPoint)


% % 原始序列的离散时间傅里叶变换DTFT
% syms w X(w);
% X(w) = 0;
% for i = 1:length(InputDots);
%     X(w) = X(w) + InputDots(i)*exp( -j*w*(i-1) );
% end
% w = [0:1:1023]*pi/512;
% Xw = double(X(w));
% figure;plot(DrawingPoint,'*')
% figure;hold on;axis([-4,4,-4,4])
% for i = 1:length(DrawingPoint)
%     plot(i,abs(DrawingPoint(i)),'*');pause(0.05);
% end