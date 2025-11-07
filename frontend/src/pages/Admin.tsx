
// Importe les bibliothèques nécessaires depuis React.
import React, { useState } from 'react';
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import togoRibbon from "@/assets/togo-ribbon.png";

// Définit un composant fonctionnel React nommé "Admin".
const Admin = () => {
  // Crée une variable d'état "file" pour stocker le fichier sélectionné par l'utilisateur.
  const [file, setFile] = useState<File | null>(null);

  // Définit une fonction qui sera appelée lorsque l'utilisateur sélectionne un fichier dans le champ de saisie.
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Vérifie si des fichiers ont bien été sélectionnés.
    if (e.target.files) {
      // Met à jour l'état "file" avec le premier fichier de la liste des fichiers sélectionnés.
      setFile(e.target.files[0]);
    }
  };

  // Définit une fonction asynchrone qui sera appelée lorsque l'utilisateur soumet le formulaire.
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    // Empêche le comportement par défaut du navigateur, qui est de recharger la page lors de la soumission d'un formulaire.
    e.preventDefault();

    // Vérifie si un fichier a été sélectionné.
    if (!file) {
      // Affiche une alerte à l'utilisateur s'il n'a pas sélectionné de fichier.
      alert('Veuillez sélectionner un fichier à envoyer.');
      // Arrête l'exécution de la fonction.
      return;
    }

    // Crée un objet FormData.
    const formData = new FormData();
    // Ajoute le fichier sélectionné à l'objet FormData avec la clé 'file'.
    formData.append('file', file);

    try {
      const apiUrl = import.meta.env.VITE_API_URL;
      // Envoie une requête POST au serveur à l'endpoint '/api/upload'.
      const response = await fetch(`${apiUrl}api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        alert('Fichier envoyé avec succès !');
      } else {
        alert("L'envoi du fichier a échoué.");
      }
    } catch (error) {
      console.error("Erreur lors de l'envoi du fichier :", error);
      alert("Une erreur s'est produite lors de l'envoi du fichier.");
    }
  };

  // La structure de la page Admin, avec le même layout que la page d'accueil.
  return (
    <div className="min-h-screen flex flex-col bg-muted/30 relative">
      <div className="absolute inset-0 bg-togo-pattern pointer-events-none" />
      <Header />
      <div className="w-full h-3 overflow-hidden">
        <img 
          src={togoRibbon} 
          alt="" 
          className="w-full h-full object-cover object-center"
        />
      </div>
      <main className="flex-1 flex flex-col items-center justify-center p-4 sm:p-6 md:p-8 relative z-10">
        <div className="bg-white p-8 rounded-lg shadow-lg w-full max-w-md">
          <h1 className="text-2xl font-bold mb-6 text-center">Admin - Envoi de PDF</h1>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="file-upload" className="block text-sm font-medium text-gray-700 mb-2">
                Sélectionner un fichier PDF
              </label>
              <input 
                id="file-upload"
                type="file" 
                accept="application/pdf" 
                onChange={handleFileChange} 
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary/10 file:text-primary hover:file:bg-primary/20"
              />
            </div>
            <button 
              type="submit"
              className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary"
            >
              Envoyer
            </button>
          </form>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Admin;
