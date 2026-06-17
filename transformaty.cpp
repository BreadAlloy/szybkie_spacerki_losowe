#include "imgui.h"

#include "transformaty.h"

#include "spacer_losowy.h"

__host__ std::string do_bin(const fp_t& X) {
	constexpr int ile_miejsca = 50;
	char buff[ile_miejsca];
	if (X == 0.0) {
		ASSERT_Z_ERROR_MSG(std::sprintf(buff, "%.1" FP_F "", X) < ile_miejsca, "Za malo miejsca w bufforze\n");
	}
	else {
		ASSERT_Z_ERROR_MSG(std::sprintf(buff, "%" FP_POS_F "", X) < ile_miejsca, "Za malo miejsca w bufforze\n");
	}
	return buff;
}

__host__ std::string do_bin(const zesp& X) {
	constexpr int ile_miejsca = 100;
	char buff[ile_miejsca];
	if (X.Re == 0.0 && X.Im == 0.0) {
		ASSERT_Z_ERROR_MSG(std::sprintf(buff, "%.1" FP_F " + i%.1" FP_F "", X.Re, X.Im) < ile_miejsca, "Za malo miejsca w bufforze\n");
	}
	else {
		if (X.Re == 0.0) {
			ASSERT_Z_ERROR_MSG(std::sprintf(buff, "%.1" FP_F " + i%" FP_POS_F "", X.Re, X.Im) < ile_miejsca, "Za malo miejsca w bufforze\n");
		}
		else {
			if (X.Im == 0.0) {
				ASSERT_Z_ERROR_MSG(std::sprintf(buff, "%" FP_POS_F " + i%.1" FP_F "", X.Re, X.Im) < ile_miejsca, "Za malo miejsca w bufforze\n");
			}
			else {
				ASSERT_Z_ERROR_MSG(std::sprintf(buff, "%" FP_POS_F " + i%" FP_POS_F "", X.Re, X.Im) < ile_miejsca, "Za malo miejsca w bufforze\n");
			}
		}
	}
	return buff;
}

template<typename towar>
__host__ void pokaz_transformate(transformata_macierz<towar>& op) {
	//ImGui::Text("Transformata:");

	ImGuiTableFlags flags = ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ContextMenuInBody;

	if (ImGui::BeginTable("cos", op.arrnosc, flags))
	{
		for (uint8_t i = 0; i < op.arrnosc; i++) {
			ImGui::TableNextRow();
			for (uint8_t j = 0; j < op.arrnosc; j++) {
				ImGui::TableSetColumnIndex(j);
				ImGui::Text(do_bin(op(i, j)).c_str());
			}
		}
		ImGui::EndTable();
	}
}

template<typename towar>
__host__ void pokaz_stan(const estetyczny_wektor<towar>& wartosci) {
	ImGui::Text("Stan:");

	ImGuiTableFlags flags = ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ContextMenuInBody;

	if (ImGui::BeginTable("Wartosci w wierzcholku", 2, flags))
	{
		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		ImGui::Text("Wartosc");
		ImGui::TableSetColumnIndex(1);
		ImGui::Text("Prawdopodobienstwo");

		for (uint64_t i = 0; i < wartosci.rozmiar; i++) {
			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);
			ImGui::Text(do_bin(wartosci[i]).c_str());
			ImGui::TableSetColumnIndex(1);
			ImGui::Text(do_bin(P(wartosci[i])).c_str());
		}
		ImGui::EndTable();
	}
}

template __host__ void pokaz_transformate(transformata_macierz<fp_t>& op);
template __host__ void pokaz_transformate(transformata_macierz<zesp>& op);

template __host__ void pokaz_stan(const estetyczny_wektor<fp_t>& wartosci);
template __host__ void pokaz_stan(const estetyczny_wektor<zesp>& wartosci);
