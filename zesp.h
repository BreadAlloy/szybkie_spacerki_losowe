#pragma once

#include "pomocne_funkcje.h"

//#ifdef __CUDA_ARCH__
//#include "zesp.cpp"
//#endif

// Nie chce u¿ywaæ funkcji z _s
#pragma warning(disable : 4996)

struct zesp {
    double Re = 0.0;
    double Im = 0.0;

    __HD__ zesp(const double Re_part = 0.0, const double Im_part = 0.0) {
        Re = Re_part;
        Im = Im_part;
    }

    __HD__ zesp(const zesp& a) {
        *this = (a);
    }

    __HD__ zesp& operator=(const zesp& a) {
        Re = a.Re;
        Im = a.Im;
        return *this;
    }

    __HD__ zesp operator-() const {
        zesp res(-Re, -Im);
        return res;
    }

    __HD__ zesp sprzezenie() const {
        zesp res(Re, -Im);
        return res;
    }

    __HD__ double norm() const {
        return Re * Re + Im * Im/* + this->y.Re * this->y.Re + this->y.Im * this->y.Im*/;
    }

    __host__ std::string str() const {
        return (std::to_string(Re) + " + j" + std::to_string(Im));
    }

    __HD__ zesp sqrt() const {
        double r = std::sqrt(Re * Re + Im * Im);
        double a = std::acos(Re / r);
        //printf("%lf, %lf, %lf\n", r , Im / r, Re / r);
        a = a / 2;
        r = std::sqrt(r);
        zesp res(std::cos(a) * r, std::sin(a) * r);
        return res;
    }

    __HD__ void operator+=(zesp const a) {
        this->Re += a.Re;
        this->Im += a.Im;
    }

    __HD__ double abs() const {
        return std::sqrt(Re * Re + Im * Im);
    }

    __HD__ static friend zesp operator+(const zesp& a, const zesp& b) {
        zesp res(a.Re + b.Re, a.Im + b.Im);
        return res;
    }

    __HD__ static friend zesp operator-(const zesp& a, const zesp& b) {
        zesp res(a.Re - b.Re, a.Im - b.Im);
        return res;
    };

    __HD__ static friend zesp operator*(const zesp& a, const zesp& b) {
        zesp res(a.Re * b.Re - a.Im * b.Im, a.Im * b.Re + a.Re * b.Im);
        return res;
    }

    __HD__ static friend zesp operator/(const zesp& a, const double b) {
        zesp res(a.Re / b, a.Im / b);
        return res;
    }

    __HD__ static friend zesp operator/(const zesp& a, const zesp& b) {
        zesp res = a - b.sprzezenie();
        res = res / (b.Re * b.Re + b.Im * b.Im);
        return res;
    }

    __HD__ static friend void operator/=(zesp& a, const zesp& b) {
        a = a / b;
    }

    __HD__ static friend void operator/=(zesp& a, const double& b) {
        a = a / b;
    }

    __HD__ static friend void operator*=(zesp& a, const zesp& b) {
        a = a * b;
    }

    __HD__ static friend void operator*=(zesp& a, const double& b) {
        a = a * b;
    }

    __HD__ static friend bool operator==(const zesp& a, const zesp& b) {
        return (a.Re == b.Re) && (a.Im == b.Im);
    }

    __HD__ static friend bool operator!=(const zesp& a, const zesp& b) {
        return (a.Re != b.Re) || (a.Im != b.Im);
    }

    __host__ static friend void z_bin(char* s, zesp& X) {
        int zwrot = std::sscanf(s, "%lf + i%lf", &(X.Re), &(X.Im));
        //printf("sscanf: %d\n", zwrot);
        if (2 != zwrot) {
            std::printf("Problem na: %s\n", s);
            ASSERT_Z_ERROR_MSG(false, "sscanf nie znalazl\n");
        }
    }
};

__HD__ zesp zero(zesp);
__HD__ zesp jeden(zesp);

