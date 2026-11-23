from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

def make_plots(path):
    p=Path(path)
    pair=pd.read_csv(p/'broad_teaching_pairs.csv')
    pair=pair.loc[pair.pair=='gray car / turtle'].copy()
    row=['R0_dinov2s','O1_dinov2b','R1_cvcl','E1_siglip','O2_openclip','E2_vitsup','E3_rn50sup','E4_mae','F1_rn50rand0','F2_pixels']
    labs=['DINOv2-small','DINOv2-base','CVCL','SigLIP','OpenCLIP','ViT supervisado','ResNet-50 supervisada','MAE','ResNet-50 aleatoria','Píxeles']
    navy='#1d3747'; teal='#007569'; muted='#526675'; orange='#ad591e'; grid='#dce4e7'
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'font.size':11,'text.color':navy,'axes.labelcolor':muted,'xtick.color':muted,'ytick.color':navy})
    df=pd.read_csv(p/'condition_by_domain.csv').query("learner=='centroid' and task=='one_word' and condition=='3sup'")
    grouped=df.groupby(['encoder','candidate_set'])[['model_fraction','human_fraction']].mean()
    def base(title,subtitle):
        f=plt.figure(figsize=(13,8),facecolor='white')
        f.text(.053,.947,'LEXICAL CONSENSUS  /  PRUEBA EXPLORATORIA CON ESTÍMULOS HUMANOS',fontsize=10,weight='bold',color=teal)
        f.text(.053,.878,title,fontsize=23,weight='bold')
        f.text(.053,.827,subtitle,fontsize=11.5,color=muted)
        return f
    def foot(f,text):
        f.add_artist(Line2D([.053,.95],[.07,.07],transform=f.transFigure,color=grid,lw=1))
        f.text(.053,.037,text,fontsize=8.6,color=muted)
    with PdfPages(p/'Agentes_y_generalizacion.pdf',metadata={'Title':'Agentes y generalización de palabras'}) as pdf:
        f=base('Qué acepta el agente tras tres ejemplos diversos','Una palabra, un centroide por palabra y una regla de aceptación basada en la dispersión de los ejemplos.')
        x0=.243; a=f.add_axes([x0,.225,.398,.49]); rows=['human']+row; colors=[teal,orange]
        for j,(k,col) in enumerate(zip(['target_superordinate','outside_domain'],colors)):
            vals=[grouped.loc[(row[0],k),'human_fraction']]+[grouped.loc[(enc,k),'model_fraction'] for enc in row]
            for i,v in enumerate(vals):
                a.scatter(100*v,i+(-.14 if j==0 else .14),s=56,marker=('o' if not j else 'D'),facecolor=col if not j else 'white',edgecolor=col,lw=1.6,zorder=3)
        a.set_yticks(range(len(rows)),['Adultos']+labs);a.set_xlim(-4,104);a.set_ylim(10.6,-.7);a.set_xticks([0,25,50,75,100]);a.set_xlabel('Imágenes aceptadas (%)',fontsize=11,labelpad=7)
        a.grid(axis='x',color=grid);a.set_axisbelow(True)
        a.axhline(.5,color=grid);a.axhline(8.5,color=grid)
        a.tick_params(length=0,pad=7)
        for sp in a.spines.values():sp.set_visible(False)
        f.legend(handles=[Line2D([],[],ls='',marker='o',color=teal,label='Superordinadas del dominio'),Line2D([],[],ls='',marker='D',mfc='white',mec=orange,label='Fuera del dominio')],loc='center left',bbox_to_anchor=(x0-.01,.76),ncol=2,frameon=False,fontsize=10,handletextpad=.18,columnspacing=.85)
        a2=f.add_axes([.705,.225,.226,.49]);a2.set_xlim(0,1);a2.set_ylim(a.get_ylim());a2.axis('off');f.text(.705,.755,'Auto gris / tortuga',weight='bold',fontsize=14)
        pcs=pair.query("task=='one_word' and learner=='centroid'").set_index('encoder')
        lab={(1,0):'Solo auto',(0,0):'Ninguno',(1,1):'Ambos',(0,1):'Solo tortuga'}
        for i,enc in enumerate(rows):
            s='458 / 600   ·   1 / 600' if enc=='human' else lab[tuple(pcs.loc[enc,['accept_A','accept_B']].astype(int))]
            a2.text(0,i,s,va='center',fontsize=11,color=navy if enc=='human' or s=='Solo auto' else muted)
        f.text(.053,.138,'En cada encoder hay 12 comparaciones superordinadas y 48 fuera del dominio. En adultos se promedian 600 sesiones por comparación.',fontsize=10.3)
        f.text(.053,.102,'La última columna se refiere solo al episodio con bus, moto y tractor. La regla de aceptación se añadió para esta prueba.',fontsize=10.3,color=muted)
        foot(f,'Fuente: Lewis y Frank (2018), XTMEM. Embeddings y aprendices de Vera, lexical-consensus, commit 23e4082.');pdf.savefig(f)
        f.savefig(p/'Agentes_y_generalizacion.png',dpi=155);plt.close(f)
        f=base('El mismo auto y la misma tortuga, con tres palabras','Cada agente aprende además palabras para animales y vegetales. Su clasificación original asigna a cada imagen una de las tres palabras.')
        q=pair.query("task=='three_words'").copy()
        q['outcome']=q.accept_A.astype(int)*2+q.accept_B.astype(int)
        cols=['centroid','kcentroid2','knn3','logreg','linsvm','random']
        mat=q.pivot(index='encoder',columns='learner',values='outcome').loc[row,cols].to_numpy()
        a=f.add_axes([.25,.216,.65,.47]);cmap=ListedColormap(['#f0f3f3','#eab38f','#b4dcd2','#d4dfec'])
        a.imshow(mat,aspect='auto',cmap=cmap,vmin=-.5,vmax=3.5)
        a.set_yticks(range(10),labs);a.set_xticks(range(6),['Centroide','2 prototipos','kNN (k=3)','Reg. logística','SVM lineal','Aleatorio'],fontsize=10)
        a.xaxis.tick_top();a.tick_params(length=0,pad=6);[sp.set_visible(False) for sp in a.spines.values()]
        for (i,j),v in np.ndenumerate(mat):a.text(j,i,['Ninguno','Tortuga','Auto','Ambos'][v],ha='center',va='center',fontsize=9.8)
        f.text(.25,.75,'Imágenes a las que el agente asigna la palabra enseñada con vehículos',fontsize=12.4,weight='bold')
        f.text(.053,.147,'Los tres dominios aportan ejemplos etiquetados. Cada celda es una decisión sobre dos imágenes para una configuración fija.',fontsize=11)
        f.text(.053,.11,'La columna aleatoria usa la semilla 42. Los resultados de las variantes adicionales están en las tablas de la corrida.',fontsize=10.3,color=muted)
        foot(f,'Fuente: salidas nativas de predict() de los seis aprendices de Lexical Consensus.');pdf.savefig(f);plt.close(f)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('outputs',type=Path);make_plots(parser.parse_args().outputs)
