#include <cstdio>
#include <stdio.h>
#include "imgIO.hpp"
#include "DCIDetect.h"
#include <vector>
#include <iostream>
//#include <wx/wx.h>
//#include "stdafx.h"
//#include "MainWindow.h"
#include <wx/wxprec.h>
#ifndef WX_PRECOMP
    #include <wx/wx.h>
#endif
void mark(HaarRectangle& rect, unsigned char* imgGrey, UInt stride)
{
	for (int i = 0; i < rect.height; ++i)
	{
		imgGrey[(rect.top + i)*stride + rect.left] = 250;
		imgGrey[(rect.top + i)*stride + rect.left + rect.width] = 250;
	}
	for (int i = 0; i < rect.width; ++i)
	{
		imgGrey[(rect.top*stride) + rect.left + i] = 250; // gora
		imgGrey[((rect.top + rect.height)*stride) + rect.left + i] = 250; //dol
	}
}
/*
int main(int argc, char *argv[])
{
	const char* pathIn = "1m.jpg";
	//const char* pathIn = "obrazTestOutColor1.jpg";
	const char* pathOutGrey = "obrazTestOutGrey.jpg";
	const char* pathOutColor = "obrazTestOutColor.jpg";

	unsigned char *imgColor = NULL;
	unsigned char *imgGrey = NULL;
	unsigned char *imgClean = NULL;

	ImgIO imgIO;
	imgIO.ReadImgColor(pathIn, imgColor);



	DCIDetect detect;

	const int imgSizeX = imgIO.getSizeX();
	const int imgSizeY = imgIO.getSizeY();

	imgIO.ColorToGray(imgColor, imgGrey);
	imgIO.ColorToGray(imgColor, imgClean);

	std::vector<HaarRectangle> result = detect.Run(imgColor, imgGrey, imgSizeX, imgSizeY);

	for (std::vector<HaarRectangle>::iterator i = result.begin(); i != result.end(); ++i)
	{
		mark(*i, imgClean, imgIO.getSizeX());
	}
	 
	imgIO.WriteImgGrey(pathOutGrey, imgClean);

	delete[] imgColor;

	//getchar();
	
	return 0;
}*/

class MyApp: public wxApp
{
public:
    virtual bool OnInit();
};
class MyFrame: public wxFrame
{
public:
    MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size);
private:
    void OnHello(wxCommandEvent& event);
	void OpenPhoto(wxCommandEvent& event);
    void OnExit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);
	void OnButtonClick(wxCommandEvent& event);
    wxDECLARE_EVENT_TABLE();
	wxImage photo;
	wxButton *button;
    wxPanel* pane;
	wxRadioBox *radiobox;
	wxRadioButton *HaarRadio;
	wxRadioButton *SkinRadio;

	 char* pathIn;
	 char* pathOutGrey;
	 char* pathOutColor;

	 unsigned char *imgColor;
	 unsigned char *imgGrey;
	 unsigned char *imgClean;

	ImgIO imgIO;

	DCIDetect *detect;
};
enum
{
    ID_Hello = 1,
	BUTTON_Hello = 2
};

//BUTTON_Hello = wxID_HIGHEST + 1;
wxBEGIN_EVENT_TABLE(MyFrame, wxFrame)
    EVT_MENU(ID_Hello,   MyFrame::OnHello)
	EVT_MENU(wxID_OPEN, MyFrame::OpenPhoto)
    EVT_MENU(wxID_EXIT,  MyFrame::OnExit)
    EVT_MENU(wxID_ABOUT, MyFrame::OnAbout)
	EVT_BUTTON(BUTTON_Hello, MyFrame::OnButtonClick)

	//EVT_PAINT(wxImagePanel::paintEvent)
wxEND_EVENT_TABLE()
wxIMPLEMENT_APP(MyApp);
bool MyApp::OnInit()
{
    MyFrame *frame = new MyFrame( "Face Detector", wxPoint(50, 50), wxSize(450, 350) );
    frame->Show( true );
    return true;
}
MyFrame::MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size)
        : wxFrame(NULL, wxID_ANY, title, pos, size)
{
	pane = new wxPanel(this, wxID_ANY);
    wxMenu *menuFile = new wxMenu;
    menuFile->Append(ID_Hello, "&Hello...\tCtrl-H",
                     "Help string shown in status bar for this menu item");
	menuFile->Append(wxID_OPEN, "Wczytaj zdjęcie");
    menuFile->AppendSeparator();
    menuFile->Append(wxID_EXIT, "Zakończ");
    wxMenu *menuHelp = new wxMenu;
    menuHelp->Append(wxID_ABOUT);
    wxMenuBar *menuBar = new wxMenuBar;
    menuBar->Append( menuFile, "&Plik" );
    menuBar->Append( menuHelp, "&Help" );
    SetMenuBar( menuBar );
    CreateStatusBar();
    SetStatusText( "Wybierz opcję z menu Plik" );
	wxInitAllImageHandlers();
	button = new wxButton(pane, BUTTON_Hello, wxT("Szukaj twarzy!"), wxPoint(100,100));
	//+radiobox->
HaarRadio = new wxRadioButton(pane, -1, 
      wxT("Algorytm obszarów kontrastowych"), wxPoint(15, 30), wxDefaultSize, wxRB_GROUP);

SkinRadio = new wxRadioButton(pane, -1, 
      wxT("Algorytm szukający kolorów i tekstur"), wxPoint(15, 55));
	//button->SetClientSize(30,30);
imgColor = NULL;
imgGrey = NULL;
imgClean = NULL;
}
void MyFrame::OnExit(wxCommandEvent& event)
{
    Close( true );
}
void MyFrame::OnAbout(wxCommandEvent& event)
{
    wxMessageBox( "This is a wxWidgets' Hello world sample",
                  "About Hello World", wxOK | wxICON_INFORMATION );
}
void MyFrame::OnHello(wxCommandEvent& event)
{
    wxLogMessage("Hello world from wxWidgets!");
}

	void MyFrame::OnButtonClick(wxCommandEvent& event) {
	if (SkinRadio->GetValue()) {
	SetStatusText("Szukam twarzy...");
	imgIO.ReadImgColor((const char*)pathIn, imgColor);
	imgIO.ColorToGray(imgColor, imgGrey);
	imgIO.ColorToGray(imgColor, imgClean);

		 int imgSizeX = imgIO.getSizeX();
	 int imgSizeY = imgIO.getSizeY();
	 detect = new DCIDetect();
		std::vector<HaarRectangle> result = detect->Run(imgColor, imgGrey, imgSizeX, imgSizeY);

	for (std::vector<HaarRectangle>::iterator i = result.begin(); i != result.end(); ++i)
	{
		mark(*i, imgClean, imgIO.getSizeX());
	}
		const char* pathGrey = "obrazTestOutGrey.jpg";
	 	//	const char* pathGrey2 = "\obrazTestOutGrey.jpg";

	imgIO.WriteImgGrey(pathGrey, imgClean);
		wxString photo_file;
	wxClientDC dc(pane);
	photo_file = pathGrey;

	//photo_file = wxGetCwd();
	//photo_file.append('\');
	//photo_file.append(wxString(pathGrey2));
	//photo_file.app
	//photo.Clear();
	dc.Clear();
	if (photo_file.find(wxString(".jpg")) != wxNOT_FOUND)
		photo.LoadFile(photo_file, "image/jpeg");
	else
	if (photo_file.find(wxString(".png")) != wxNOT_FOUND)
		photo.LoadFile(photo_file, "image/png");
	else
	if (photo_file.find(wxString(".gif")) != wxNOT_FOUND)
		photo.LoadFile(photo_file, "image/gif");
		//dc.DrawBitmap(photo,10,10,false);
		dc.DrawBitmap(photo, (GetClientSize().GetX() - photo.GetWidth()) / 2, 10, false);

	SetStatusText("Zakończono operację.");

	}
	}

void MyFrame::OpenPhoto(wxCommandEvent& event) {

	wxString photo_file;
	wxClientDC dc(pane);
	wxFileDialog wxfd(this, ("Wybierz fotografię"), "", "", "Pliki (*.gif, *.jpg, *.png)|*jpg;*png;*gif", wxFD_OPEN | wxFD_FILE_MUST_EXIST);
	if (wxfd.ShowModal() == wxID_OK) {
	//SetStatusText(wxfd.GetPath());
	photo_file = wxfd.GetPath();
	
	//photo.Clear();
	dc.Clear();
	if (photo_file.find(wxString(".jpg")) != wxNOT_FOUND)
		photo.LoadFile(photo_file, "image/jpeg");
	else
	if (photo_file.find(wxString(".png")) != wxNOT_FOUND)
		photo.LoadFile(photo_file, "image/png");
	else
	if (photo_file.find(wxString(".gif")) != wxNOT_FOUND)
		photo.LoadFile(photo_file, "image/gif");
//	photo_area.DrawBitmap(photo,0,0, false);
	///wxFilterInputStream picture
	photo.GetHeight();
	photo.GetWidth();
	//this->set
	if (photo.GetWidth() >= 380)
		SetClientSize(photo.GetWidth()+20, photo.GetHeight()+60);
	else
		SetClientSize(400, photo.GetHeight()+60);

	button->SetPosition(wxPoint(10, photo.GetHeight()+10));
	HaarRadio->SetPosition(wxPoint(110, photo.GetHeight()+10));
	SkinRadio->SetPosition(wxPoint(110, photo.GetHeight()+35));
	
	dc.DrawBitmap(photo, (GetClientSize().GetX() - photo.GetWidth()) / 2, 10, false);
	//dc.DrawBitmap(photo,10,10,false);
	std::string pathIns = photo_file.ToStdString();
	pathIn = new char[pathIns.length()+1];
	strcpy(pathIn, pathIns.c_str());
	pathIn[pathIns.length()] = '\0';
	
	imgColor = NULL;
imgGrey = NULL;
imgClean = NULL;
	//Fit();
	//SetStatusText(	photo.GetWidth());
	}
}
