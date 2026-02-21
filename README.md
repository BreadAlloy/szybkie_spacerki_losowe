## Szybke spacerki losowe
W tym stanie nie bêdzie siê kompilowæ na innych systemach ni¿ moim.
Gdyby komuœ zale¿a³o na zkompliowaniu u siebie to proszê o kontakt. 

<p align="center">
<video
muted 
loop 
preload="auto" 
autoPlay 
playsInline 
width=500
src="obrazy/dwu_okrag_ewolucja.mp4">

</video>
</p>

<p align="center">
  <img src="obrazy/przyklady_ewolucji.png" width=1000 height=/>
</p>

## [Algorytm szybkiego spacerku losowego](./algorytm_szybkiego_spacerku_losowego.md)
Du¿o bawi³em siê w zrobienie tego, aby ewolucja spaceru losowego by³a szybka.
I uwa¿am ¿e, zrobienie coœ znacz¹co szybszego ni¿ ja zrobi³em, bêdzie trudne, jeœli nie niemo¿liwe.

## Ogólne informacje
### Kolejnoœæ operacji w iteracji spaceru
1. Mieszanie w wierzcho³ku(operator monety)
2. Wys³anie do docelowych kube³ków(operator przesuniêcia)
3. Absorbcja z podanych kub³eków(opcjonalne usuniêcie prawdopodobieñstwa)

### Ró¿ne operatory monety
Fizycy lubi¹ u¿ywaæ jednego operatora monety na wszystkich wierzcho³kach. Mog¹ te¿ wtedy, zapisaæ globalny operator jako, $iloczynkronekera(operator perzesuniecia, operator monety)$.
Natomiast przestaje to dzia³aæ, gdy jakiœ wierzcho³ek jest innego stopnia.
Dlatego w mojej implementacji ka¿dy wierzcho³ek ma swój w³asny operator monety.
Mo¿e to byæ ten sam operator w ka¿dym, jeœli ka¿dy wierzcho³ek ma ten sam stopieñ.

### Inny operator przesuniêcia ni¿ normalny
Grafy na których licze rzeczy trochê siê ró¿ni¹ operatorem przesuniêcia.
Klasyczna definicja operatora przesuniêcia nie dzia³a dobrze dla dowolnego grafu. 
W przypadku dla lini fizycy by chcieli:

<p align="center">
  <img src="obrazy/uklad_fizykow.png" width=600 height=/>
</p>

Wtedy dla mieszacza prawdopodobieñstwa identycznoœciowego, prawdopodobieñstwo przemieszcza siê, ze ze sta³a prêdkoœci¹.
Ja natomiast u¿ywam takiego grafu:

<p align="center">
  <img src="obrazy/uklad_moj_skier.png" width=600 height=/>
</p>

Teraz dla mieszacza identycznoœciowego, prawdopodobieñstwo prawie, ¿e stoi w miejscu.
I mogê graf przedstawiæ za pomoc¹ krawêdzi nieskierowanych:

<p align="center">
  <img src="obrazy/uklad_moj_nieskier.png" width=600 height=/>
</p>

Obydwie definicje operatora mog¹ robiæ to samo, przy odpowiedniej permutacji wejœæ lub/i wyjœæ funkcji mieszania prawdopodobieñstwa na wierzcho³kach.
Konwencja, ¿e kierunek prawdopodobieñstwa ma byæ zachowany, nie dzia³a dobrze dla dowolnego grafu. WeŸmy za przyk³ad graf:

<p align="center">
  <img src="obrazy/zagadka.png" width=300 height=/>
</p>

I pytanie jest jak zdefiniowaæ przejœcia prawdopodobieñstwa miêdzy kube³kami, aby "zachowywa³o swój kierunek"?

Uwa¿am, ¿e nie istnieje algorytm rozwi¹zuj¹cy to zagadnienie dla dowolnego grafu. Proponuje siê nie zastanawiaæ nad tym. Po prostu daæ krawêdzie nieskierowane i zostawiæ takie rozwa¿ania do operatora monety na wierzcho³ku.
<p align="center">
  <img src="obrazy/zagadka_rozwiazanie.png" width=300 height=/>
</p>

### Kolejnoœæ argumentów mieszacza prawdopodobieñstwa
Definicja po³¹czeñ miêdzy kube³kami wp³ywa na kolejnoœæ argumentów. WeŸmy przypadek dla macierzy.
Zdarzy³o siê, ¿e zastanawia³em siê dlaczego mam inne wyniki ni¿ kolega.
Okaza³o siê, ¿e macierz operatora monety dostawa³a i zwraca³a inn¹ permutacje wartoœci w kube³kach.
Wiêc jest to wa¿ne. U mnie na wierzcho³ku dzia³a to nastêpuj¹co:
<p align="center">
  <img src="obrazy/kolejnosc_w_wektorze.png" width=300 height=/>
</p>


## Uwagi nazewnictwowe:
- kube³ek i kierunek oznaczaj¹ praktycznie to samo
- transformer fizycy nazwali by operatorem, albo operatorem monety, ale operator jest zarezerwowanym keywordem w c++, wiêc u¿ywam nazwy transformer, albo transformata
- towar to jest zawartoœæ kube³ka
- template typ zawartoœci kube³ka nazywa siê towar
- template typ operatora mieszaj¹cego nazywa siê transformata


## Dla tych co c++ nie znaj¹, ale kod chc¹ poczytaæ:
Zak³adam, ¿e Pythona znasz. c++ nie ró¿ni siê du¿o sk³adniowo od Pythona.
Poni¿ej zapisa³em podobieñstwa i ró¿nice miêdzy jêzykami które mam nadzieje pozowl¹ ci zrozumieæ kod na podstawie znajomoœci Pythona.

### \{\} zamiast wciêæ po : 

### Wszêdzie na koñcu jest ;
W Pythonie te¿ tak mo¿na, tu jednak jest to konieczne.

### Wszêdzie jest sta³y typ, poza miejscami gdzie typ jest "zmienny"
Zmienne w c++ maj¹ zdefiniowany sta³y typ. Jest jednak mechanizm operowania na typach, tak ¿e udaje zmienne typy.
Nazywa siê on template. Podczas kompilacji i tak s¹ sta³e typy. Rzeczy zwi¹zane z tym mechanizmem s¹ w nawiasie <>.
Silnie z niego korzystam w structcie spacer_losowy, aby na raz móg³ obs³ugiwaæ najwiêcej typów spacerów losowych na podstawie argumentów templateowych.
Template zawartoœci kube³ka nazywa siê towar. Template operatora mieszaj¹cego nazywa siê transformata. 

### Troche inne fory
| Python | c++ |
| :-: | :-: |
| for i in X | for(auto& i : X) |
| for i in range(n, m, k) | for(int i = n; i < m; i += k) |

## Szczegó³y techniczne projektu
Warningi ze wszystkich plików Ÿród³owych z grafiki komputerowej s¹ ignorowane(zrobione w solution explorer albo/i #pragma). I wszystkie warningi w plikach cpp w folderze s¹ zdisablowane.



