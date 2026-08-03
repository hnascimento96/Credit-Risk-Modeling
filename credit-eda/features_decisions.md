Tratamento de missing:

'Income': imputação pela mediana, mais robusta aos outliers.
'No.Dep.': imputação pela moda, uma vez que a variável é inteira (discreta).

Considerando a baixa diferença percentual entre inadimplentes não-nulos e inadimplentes nulos das variáveis,
não fez sentido criação de flag missing para nenhuma delas.

__________________________________________________________________________

Tratamento de outliers

A fim de se verificar o desempenho, em especial da regressão logística, o tratamento de outliers foi adiado
pra se comparar o modelo tratado com o modelo cru (raw).

__________________________________________________________________________
Feature creation

AnyLate: flag que indica se houve algum atraso (1 - sim, 0 - não)
TotalLate: variável que indica o total de atrasos
LateScore: variável de score que atribui uma pontuação ponderada para os atrasos: 30-59, peso 1; 60-89, peso 2; 90+: peso 3
UtilAbove100: flag que indica se a utilização do limite de crédito excedeu 100%.