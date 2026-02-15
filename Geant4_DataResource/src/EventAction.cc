//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
/// \file B4/B4a/src/EventAction.cc
/// \brief Implementation of the B4a::EventAction class

#include "EventAction.hh"

#include <fstream>
#include "G4AnalysisManager.hh"
#include "G4Event.hh"
#include "G4RunManager.hh"
#include "G4UnitsTable.hh"

#include <iomanip>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <sstream>

namespace B4a
{

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

  EventAction::EventAction()
  : G4UserEventAction(),
  fSocket(-1),
  fConnected(false)
  {
    // Initialize our vector with 10 zeros (assuming 10 layers)
    fLayerEnergies.assign(10, 0.0);

    // --- CONNECT TO SERVER ---
    // We attempt to connect to localhost port 5000 (We will build a listener here later)

    fSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (fSocket < 0) {
      G4cerr << "Error creating socket!" << G4endl;
      return;
    }

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(5001); // The port our Java Bridge will listen on

    // Convert IPv4 address from text to binary form
    // "127.0.0.1" is localhost. If you run Docker, this connects to your Mac host network.
    if(inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr)<=0) {
      G4cerr << "Invalid address/ Address not supported" << G4endl;
      return;
    }

    if (connect(fSocket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
      G4cout << "⚠️ CONNECTION FAILED: Simulation will run without streaming." << G4endl;
      fConnected = false;
    } else {
      G4cout << "✅ CONNECTED TO DATA PIPELINE SERVER!" << G4endl;
      fConnected = true;
    }
  }

  EventAction::~EventAction()
  {
    if (fConnected && fSocket >= 0) {
      close(fSocket);
    }

  }

void EventAction::BeginOfEventAction(const G4Event* /*event*/)
{
  // initialisation per event
  fEnergyAbs = 0.;
  fEnergyGap = 0.;
  fTrackLAbs = 0.;
  fTrackLGap = 0.;

  std::fill(fLayerEnergies.begin(), fLayerEnergies.end(), 0.0);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::EndOfEventAction(const G4Event* event)
{
  // get analysis manager
  auto analysisManager = G4AnalysisManager::Instance();

  // fill histograms
  analysisManager->FillH1(0, fEnergyAbs);
  analysisManager->FillH1(1, fEnergyGap);
  analysisManager->FillH1(2, fTrackLAbs);
  analysisManager->FillH1(3, fTrackLGap);

  // fill ntuple
  analysisManager->FillNtupleDColumn(0, fEnergyAbs);
  analysisManager->FillNtupleDColumn(1, fEnergyGap);
  analysisManager->FillNtupleDColumn(2, fTrackLAbs);
  analysisManager->FillNtupleDColumn(3, fTrackLGap);
  analysisManager->AddNtupleRow();


  //  Identify the Particle
  // 0 = Electron (e-)
  // 1 = Pion (pi-)
  // 2 = Muon (mu-)
  // 3 = Gamma (gamma)
  G4int label = 0; // Default to 0 (Electron)

  // Get the name of the particle that started the event
  G4String particleName = event->GetPrimaryVertex(0)->GetPrimary()->GetParticleDefinition()->GetParticleName();
    if (particleName == "e-") label = 0;
    else if (particleName == "pi-") label = 1;
    else if (particleName == "mu-") label = 2;
    else if (particleName == "gamma") label = 3;

    // 2. Format the Payload
    // Format: "ID,Label,E1,E2,E3,E4,E5,E6,E7,E8,E9,E10\n"
    std::ostringstream msg;
    msg << event->GetEventID() << "," << label;

    for (double energy : fLayerEnergies) {
      msg << "," << energy;
    }
    msg << "\n"; // Newline indicates end of message

    std::string payload = msg.str();

    // 3. Send over Network
    if (fConnected) {
      send(fSocket, payload.c_str(), payload.length(), 0);
    }








  if (particleName == "e-") {
    label = 0;
  }
  else if (particleName == "pi-") {
    label = 1;
  }
  else if (particleName == "mu-") {
    label = 2;
  }
  else if (particleName == "gamma") {
    label = 3;
  }

  // 2. Open the CSV file in "Append" mode (add to bottom)
  // Note: In a real high-performance app, we'd keep this open in RunAction,
  // but opening it here is safer for a beginner to avoid crashes.
  std::ofstream csvFile("training_data.csv", std::ios::app);

  if (csvFile.is_open()) {
    // Write the Label first
    csvFile << label;

    // Write the 10 Layer Energies
    for (size_t i = 0; i < fLayerEnergies.size(); i++) {
      csvFile << "," << fLayerEnergies[i];
    }

    // End the row
    csvFile << "\n";
    csvFile.close();
  }




  // Print per event (modulo n) only after it is sent over the network
  //
  auto eventID = event->GetEventID();
  auto printModulo = G4RunManager::GetRunManager()->GetPrintProgress();
  if ((printModulo > 0) && (eventID % printModulo == 0)) {
    G4cout << "   Absorber: total energy: " << std::setw(7) << G4BestUnit(fEnergyAbs, "Energy")
           << "       total track length: " << std::setw(7) << G4BestUnit(fTrackLAbs, "Length")
           << G4endl << "        Gap: total energy: " << std::setw(7)
           << G4BestUnit(fEnergyGap, "Energy") << "       total track length: " << std::setw(7)
           << G4BestUnit(fTrackLGap, "Length") << G4endl;

    G4cout << "--> End of event " << eventID << "\n" << G4endl;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

  void EventAction::AddEnergy(G4int layerIndex, G4double energy)
{
  // Safety check: Don't crash if index is weird
  if (layerIndex >= 0 && layerIndex < fLayerEnergies.size()) {
    fLayerEnergies[layerIndex] += energy;
  }
}

}  // namespace B4a
