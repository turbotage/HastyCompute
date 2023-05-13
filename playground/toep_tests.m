
n = 3;

t = complex(rand(n,1),rand(n,1));

T = toeplitz(t);
%T(1:n+1:end) = abs(T(1:n+1:end));

C = zeros(2*n,2*n);

B = zeros(n,n);
for i=1:n
    B = B + diag(T(n+1-i)*ones(n-i,1),i);
end
B = B + B';

C(1:n,1:n) = T;
C((n+1):end,(n+1):end) = T;
C((n+1):end,1:n) = B;
C(1:n,(n+1):end) = B;

disp(T);
disp(C);

x = rand(n,1);

y = T * x;

%F = dftmtx(2*n);
dfft = diag(F * C * F' / (2*n))
yn = ifft(dfft.*fft(x,2*n));
yn = yn(1:n);



disp(y);
disp(yn);


