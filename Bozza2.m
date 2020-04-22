%Estrazione del dataset di immagini e delle labels corrispondenti
k = 1;
n = 1;
u = 1;
v = 1;
filepath1 = "Dataset\disc" + num2str(n);
filepath2 = "\OAS1_000";
filepath3 = "_MR1\OAS1_000";
filepath4 = "_MR1.txt";

i = 1;

for z = 1 : numel(dir(fullfile("Dataset\","disc*"))) %Scorre i dischi
    for j = 1 :numel(dir(fullfile(filepath1,"OAS1_*"))) %Scorre le cartelle nei dischi
    
        if i > 9 
            filepath2 = "\OAS1_00";
            filepath3 = "_MR1\OAS1_00";
        end
        if i > 99
            filepath2 = "\OAS1_0";
            filepath3 = "_MR1\OAS1_0";
        end
        %Genero il path del file di testo
        filepath = filepath1 + filepath2 + num2str(i)+ filepath3 + num2str(i) + filepath4;
        R = filepath1 + filepath2 + num2str(i) + "_MR1\RAW"; %Path della cartella RAW
        S = dir(fullfile(R,'*.gif'));
        array = strings(1,numel(S)); %Inizializzo l'array
        CDR = parseSubjectStatus(filepath); %Ricavo il Clinical Dementia Rating (livello di demenza)
          if CDR == 0 %Se � nullo, il paziente � sano
              healthyIndx(u) = CDR; %Salvo l'indice
              %Salvo le immagini in un cell array   
              for k = 1: numel(S)
                 array(k) = fullfile(R,S(k).name);
              end
              healthyImgs{u} = array;
              u=u+1;
          end
        if CDR > 0 %Se � > 0, il paziente � affetto da demenza
            i
            dementIndx(v) = CDR; % Salvo l'indice
            %Salvo le immagini in un cell array 
            for k = 1: numel(S)
                array(k) = fullfile(R,S(k).name);                
            end
            dementImgs{v} = array;
            v = v+1
        end     
    i = i+1;
    end   
    n = n+1;
    filepath1 = "Dataset\disc" + num2str(n)
end

