#pragma once

#include "zesp.h"

__HD__ zesp zero(zesp) {
	return zesp(0.0, 0.0);
}

__HD__ zesp jeden(zesp) {
	return zesp(1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0));
}