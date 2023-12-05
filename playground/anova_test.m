
datatransp = zeros(3*27,7);

for i=1:7
    
    for j=1:3
        istart = (i-1)*27 + 1;
        col = data(istart:istart+26,j);
        jstart = (j-1)*27 + 1;
        datatransp(jstart:jstart+26,i) = col;
    end

end


