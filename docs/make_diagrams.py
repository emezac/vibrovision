#!/usr/bin/env python3
"""Regenerate diagrams of the implementation and verified AWS demo."""
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE=Path(__file__).resolve().parent
COLORS=['#2563a8','#16816c','#7754a1','#bd641e']

def box(ax,x,y,w,h,title,body,color=0):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',fc='#f7f9fc',ec=COLORS[color],lw=1.5))
    ax.text(x+.015,y+h-.025,title,va='top',fontsize=11,weight='bold',color=COLORS[color])
    ax.text(x+.015,y+h-.075,body,va='top',fontsize=9.5,linespacing=1.5,color='#263340')

def arrow(ax,a,b,rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,lw=1.4,color='#5d6975',connectionstyle=f'arc3,rad={rad}'))

def canvas(title):
    fig,ax=plt.subplots(figsize=(13,8),dpi=150)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.02,.97,title,fontsize=17,weight='bold',va='top')
    return fig,ax

def architecture():
    fig,ax=canvas('VibroVision 3 · verified AWS demo · us-east-2')
    box(ax,.02,.71,.27,.17,'Browser / FastAPI UI','Video input and evidence review\nAPI key required for analysis')
    box(ax,.36,.71,.27,.17,'CloudFront','Browser HTTPS endpoint\nOrigin header verification',1)
    box(ax,.70,.71,.28,.17,'EC2 Graviton / Docker','t4g.small · aarch64\nOpenCV 5.0.0 · one worker',1)
    arrow(ax,(.30,.79),(.35,.79));arrow(ax,(.64,.79),(.69,.79))
    box(ax,.02,.40,.27,.20,'OpenCV measurement','Regions and displacement\nCamera compensation\nQuality and independent checks',1)
    box(ax,.36,.40,.27,.20,'Spectral / physical checks','Alpha and joint spectra\nConditional mechanical model\nEmpirical uncertainty',1)
    box(ax,.70,.40,.28,.20,'Agent + human review','Adapt the next measurement\nGenerate evidence or abstain\nWait for human approval',2)
    arrow(ax,(.84,.70),(.16,.61),rad=.12)
    arrow(ax,(.30,.50),(.35,.50));arrow(ax,(.64,.50),(.69,.50))
    arrow(ax,(.71,.39),(.18,.39),rad=-.28)
    ax.text(.40,.31,'measurement feedback',fontsize=9,color=COLORS[2])
    box(ax,.02,.09,.27,.16,'Private S3','Encrypted analysis evidence\n30-day evidence lifecycle',0)
    box(ax,.36,.09,.27,.16,'CloudWatch','Logs and embedded metrics\n14-day log retention',0)
    box(ax,.70,.09,.28,.16,'Secrets Manager / role','Generated access key\nScoped instance permissions',3)
    ax.text(.02,.025,'Browser → CloudFront: HTTPS. Restricted CloudFront → EC2: HTTP. Stock pip OpenCV; no COOL speedup claim.',fontsize=8.5,color='#465564')
    fig.savefig(HERE/'architecture.png',bbox_inches='tight',facecolor='white');plt.close(fig)

def agent_flow():
    fig,ax=canvas('La evidencia cambia lo que hace el agente')
    boxes=[(.03,.69,'Medir','Textura, movimiento y espectro'),(.37,.69,'Comprobar','Calidad, modelo e incertidumbre'),(.71,.69,'Decidir','¿Qué permite concluir el video?')]
    for x,y,t,b in boxes:box(ax,x,y,.26,.18,t,b,0)
    arrow(ax,(.30,.78),(.36,.78));arrow(ax,(.64,.78),(.70,.78))
    box(ax,.03,.36,.26,.21,'Información insuficiente','Reubicar región o ampliar ventana\nPedir un clip mejor o más largo\nAbstenerse de localizar',2)
    box(ax,.37,.36,.26,.21,'Cambio sin localización','Conservar evidencia del cambio\nNo señalar una pieza por intuición\nSolicitar otra medición',2)
    box(ax,.71,.36,.26,.21,'Cambio localizado','Confirmar temporalmente\nMostrar evidencia al ingeniero\nEsperar aprobación',3)
    arrow(ax,(.78,.68),(.16,.58),rad=.2);arrow(ax,(.81,.68),(.50,.58));arrow(ax,(.84,.68),(.84,.58))
    box(ax,.03,.06,.94,.17,'Alcance de las garantías',
        'Identidades exactas bajo hipótesis explícitas; incertidumbre bootstrap empírica, sin cobertura certificada.\n'
        'Los clips reales actuales permiten comparación de movimiento; no reconstruyen todos los anclajes.',1)
    fig.savefig(HERE/'agent_workflow.png',bbox_inches='tight',facecolor='white');plt.close(fig)

def mechanical_core():
    fig,ax=canvas('Qué añade la matemática al análisis del video')
    box(ax,.03,.66,.44,.21,'Escala aparente distinta','Las regiones pueden verse a tamaños diferentes.\nEl operador conserva las propiedades mecánicas\nbajo el modelo de masas y resortes recíprocos.',1)
    box(ax,.53,.66,.44,.21,'Identidad que lo permite','B = S M⁻¹ K S⁻¹\nDᵢᵢ = Bᵢᵢ ; Dᵢⱼ = −√(Bᵢⱼ Bⱼᵢ)\nD = M⁻¹ᐟ² K M⁻¹ᐟ²',0)
    box(ax,.03,.35,.44,.21,'Qué movimiento ve cada defecto','Una unión se reconoce mejor en modos donde\nsus extremos se mueven de forma diferente.\nUna fórmula exacta describe la flexibilidad.',1)
    box(ax,.53,.35,.44,.21,'Cuándo no concluir','Si faltan modos o domina la incertidumbre,\nno se nombra un soporte dañado.\nSe pide una mejor medición.',2)
    box(ax,.03,.05,.94,.17,'Tres cantidades, tres preguntas',
        'Rigidez: frecuencias naturales.   Amortiguamiento: cuánto tarda en desaparecer el movimiento.\n'
        'Conectividad: qué grupos están débilmente unidos. Una sola cifra no responde todo.',3)
    fig.savefig(HERE/'jacobians.png',bbox_inches='tight',facecolor='white');plt.close(fig)

if __name__=='__main__':
    architecture();agent_flow();mechanical_core();print('Current diagrams generated')
