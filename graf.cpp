#include "graf.h"

graf graf_lini(uint32_t liczba_wierzcholkow, bool z_nazwami){
	/*               -------=== G ===--------
			|0}	|1} |2} |3}         |n-2}|n-1}

			 0  /0  /0  /0   ( ... )  /0  /0
			   /   /   /   /         /   /
			 1/  1/  1/  1/			   1/  1

			 n = liczba_wierzcholkow
	*/
	graf G(liczba_wierzcholkow);

	for (ID_W i = 0; i < liczba_wierzcholkow; i++) {
		G.zdefiniuj_liczbe_polaczen(i, 2);
	}

	if(z_nazwami){
		char temp[1000]; 
		for(ID_W i = 0; i < liczba_wierzcholkow; i++) {
			int size = snprintf(temp, 1000, "|%d}", i);
			G.update_opis(i, std::string(temp, size));
		}
	}

	G.dodaj_krawedz(0, 0);
	for (ID_W i = 1; i < liczba_wierzcholkow; i++) {
		G.dodaj_krawedz_nieskier(i - 1, i);
	}
	G.dodaj_krawedz(liczba_wierzcholkow - 1U, liczba_wierzcholkow - 1U);

	ASSERT_Z_ERROR_MSG(G.czy_gotowy(), "w grafie lini sa brakujace polaczenia\n");
	return G;
}

graf graf_lini_cykl(uint32_t liczba_wierzcholkow, bool z_nazwami){
	/*           -------=== G ===--------
			|0}	|1} |2} |3}         |n-2}|n-1}

		    /0  /0  /0  /0   ( ... )  /0  /0
		   /   /   /   /   /         /   /
 		   | 1/	 1/  1/  1/			   1/  1
           |                               |
		   |-------------------------------|	   
			 n = liczba_wierzcholkow
	*/
	graf G(liczba_wierzcholkow);

	for (ID_W i = 0; i < liczba_wierzcholkow; i++) {
		G.zdefiniuj_liczbe_polaczen(i, 2);
	}

	if (z_nazwami) {
		char temp[1000];
		for (ID_W i = 0; i < liczba_wierzcholkow; i++) {
			int size = snprintf(temp, 1000, "|%d}", i);
			G.update_opis(i, std::string(temp, size));
		}
	}

	for (ID_W i = 1; i < liczba_wierzcholkow; i++) {
		G.dodaj_krawedz_nieskier(i - 1, 1, i, 0);
	}
	G.dodaj_krawedz_nieskier(liczba_wierzcholkow - 1, 1, 0, 0);

	ASSERT_Z_ERROR_MSG(G.czy_gotowy(), "w grafie lini cykl sa brakujace polaczenia\n");
	return G;
}

graf graf_krata_2D(uint32_t liczba_wierzcholkow_boku, bool z_nazwami){
	/*			i(x)
		+------->
		|	  2			   2			2				   2			  2
		| 0 |0,0} 1 -- 0 |1,0} 1 -- 0 |2,0} 1 -(...)- 0 |n-2,0} 1 -- 0 |n-1,0} 1
		|     3			   3            3				   3		      3
		|	  |			   |			|				   |		  	  |
	j(y)V	  2			   2			2		           2              2
		  0 |0,1} 1 -- 0 |1,1} 1 -- 0 |2,1} 1 -(...)- 0 |n-2,1} 1 -- 0 |n-1,1} 1
			  3 		   3			3				   3		      3
			  |			   |			|				   |			  |
			(...)        (...)  	  (...)              (...)  		(...)
			  |			   |			|				   |			  |
			  2  		   2	        2  				   2              2		  	
		 0 |0,n-2} 1--0 |1,n-2} 1--0 |2,n-2} 1-(...)-0 |n-2,n-2} 1--0 |n-1,n-2} 1
			  3			   3			3   			   3 			  3 
			  |			   |            |				   |			  |
			  2		       2     		2	  	 	       2              2
		 0 |0,n-1} 1--0 |1,n-1} 1--0 |2,n-1} 1-(...)-0 |n-2,n-1} 1--0 |n-1,n-1} 1
			  3			   3			3				   3			  3

			n = liczba_wierzcholkow_boku
	*/
	graf linia(graf_lini(liczba_wierzcholkow_boku, BEZ_NAZW));
	graf G(0);
	
	for(uint32_t i = 0; i < linia.liczba_wierzcholkow(); i++){
		linia.dodaj_polaczenia(i, 2);
	}

	for(uint32_t j = 0; j < liczba_wierzcholkow_boku; j++){
		G += linia;

		if (z_nazwami) {
			char temp[1000];
			for(ID_W i = G.liczba_wierzcholkow() - liczba_wierzcholkow_boku; i < G.liczba_wierzcholkow(); i++){
				int size = snprintf(temp, 1000, "|%d:%d}", i % liczba_wierzcholkow_boku, j);
				G.update_opis(i, std::string(temp, size));
			}
		}
	}

	for (uint32_t i = 0; i < liczba_wierzcholkow_boku; i++) {
		if (i == 0) {
			for (uint32_t j = 0; j < liczba_wierzcholkow_boku; j++) {
				G.dodaj_krawedz(j, j);
			}
		}

		for (uint32_t j = 1; j < liczba_wierzcholkow_boku; j++) {
			G.dodaj_krawedz_nieskier(i + (liczba_wierzcholkow_boku) * j, i + liczba_wierzcholkow_boku*(j - 1));
		}

		if (i == liczba_wierzcholkow_boku - 1U) {
			for (uint32_t j = liczba_wierzcholkow_boku * i; j < liczba_wierzcholkow_boku * liczba_wierzcholkow_boku; j++) {
				G.dodaj_krawedz(j, j);
			}
		}
	}

	ASSERT_Z_ERROR_MSG(G.czy_gotowy(), "w grafie kraty 2D sa brakujace polaczenia\n");
	return G;
}

graf graf_krata_2D_cykl(uint32_t liczba_wierzcholkow_boku, bool z_nazwami) {
	/*			i(x)
		+------->
		|		|			 |			  |					 |				|
		|	    2			 2			  2				     2			    2
		| - 0 |0,0} 1 -- 0 |1,0} 1 -- 0 |2,0} 1 -(...)- 0 |n-2,0} 1 -- 0 |n-1,0} 1 -
		|       3			 3            3				     3		        3
		|	    |			 |			  |				     |		  	    |
	j(y)V	    2			 2			  2		             2              2
		  - 0 |0,1} 1 -- 0 |1,1} 1 -- 0 |2,1} 1 -(...)- 0 |n-2,1} 1 -- 0 |n-1,1} 1 -
			    3 		     3		      3				     3		        3 
		        |			 |			  |				     |			    |
			  (...)        (...)  	    (...)              (...)  		  (...)
		        |			 |			  |				     |			    |
			    2  		     2	          2  				 2              2
		 - 0 |0,n-2} 1--0 |1,n-2} 1--0 |2,n-2} 1-(...)-0 |n-2,n-2} 1--0 |n-1,n-2} 1 -
			    3			 3		      3   			     3 			    3
			    |			 |            |				     |		    	|
			    2		     2     		  2	  	 	         2              2
		 - 0 |0,n-1} 1--0 |1,n-1} 1--0 |2,n-1} 1-(...)-0 |n-2,n-1} 1--0 |n-1,n-1} 1 -
			    3			 3			  3				     3			    3
				|			 |			  |					 |				|

			n = liczba_wierzcholkow_boku
	*/
	graf linia(graf_lini_cykl(liczba_wierzcholkow_boku, BEZ_NAZW));
	graf G(0);

	for (uint32_t i = 0; i < linia.liczba_wierzcholkow(); i++) {
		linia.dodaj_polaczenia(i, 2);
	}

	for (uint32_t j = 0; j < liczba_wierzcholkow_boku; j++) {
		G += linia;

		if (z_nazwami) {
			char temp[1000];
			for (ID_W i = G.liczba_wierzcholkow() - liczba_wierzcholkow_boku; i < G.liczba_wierzcholkow(); i++) {
				int size = snprintf(temp, 1000, "|%d:%d}", i % liczba_wierzcholkow_boku, j);
				G.update_opis(i, std::string(temp, size));
			}
		}
	}

	for (uint32_t i = 0; i < liczba_wierzcholkow_boku; i++) {
		G.dodaj_krawedz_nieskier(i, 2, liczba_wierzcholkow_boku * (liczba_wierzcholkow_boku - 1) + i, 3);
	}

	for (uint32_t i = 0; i < liczba_wierzcholkow_boku; i++) {
		for (uint32_t j = 1; j < liczba_wierzcholkow_boku; j++) {
			G.dodaj_krawedz_nieskier(i + (liczba_wierzcholkow_boku)*j, i + liczba_wierzcholkow_boku * (j - 1));
		}
	}

	ASSERT_Z_ERROR_MSG(G.czy_gotowy(), "w grafie kraty 2D sa brakujace polaczenia\n");
	return G;
}

graf graf_krata_2D_z_przekatnymi(uint32_t liczba_wierzcholkow_boku, bool z_nazwami){
	graf G;
	ASSERT_Z_ERROR_MSG(G.czy_gotowy(), "w grafie kraty z przekatnymi sa brakujace polaczenia\n");
	return G;
}

graf graf_krata_3D(uint32_t liczba_wierzcholkow_boku, bool z_nazwami){
	graf G;
	ASSERT_Z_ERROR_MSG(G.czy_gotowy(), "w grafie kraty 3D sa brakujace polaczenia\n");
	return G;
}


void test_funkcji_tworzacych_grafy(){
	graf linia3 = graf_lini(3);
	graf linia10 = graf_lini(10);
	graf linia1000 = graf_lini(1000);
	graf cykl3 = graf_lini_cykl(3);
	graf cykl10 = graf_lini_cykl(10);
	graf cykl1000 = graf_lini_cykl(1000);
	graf krata2d_3 = graf_krata_2D(3);
	graf krata2d_10 = graf_krata_2D(10);
	graf krata2d_1000 = graf_krata_2D(1000);
}