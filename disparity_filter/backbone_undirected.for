      implicit double precision(w,s,d)
      character*80 filename
      character*80 filenameout

      parameter (NODOSMAX=100000,NEDGESMAX=1000000)



      dimension internalnet(1:NEDGESMAX,1:2)
      dimension weight(1:NEDGESMAX)
      dimension strenght(1:NODOSMAX)
      dimension ndegree(1:NODOSMAX)

      weightmin=0.
      wconfidence=0.9977   !this is 1-alfa

      filename='../Data/distancesbetweencities_Spain_ids.csv'  !input network
      filenameout='../Data/backbone_Spain_0.9977_nw.net'                   !output backbone

      do i=1,NODOSMAX
        strenght(i)=0.
        ndegree(i)=0
      enddo

      do i=1,NEDGESMAX
      weight(i)=0.
      enddo

*      write(*,*) 'This program filters out the backbone'
*      write(*,*) 'of a weighted undirected network, format links not repeated'
*      write(*,*) 'gives a network, the backbone'
*      write(*,*) 'Maximum number of nodes=',NODOSMAX
*      write(*,*) 'Maximum number of links=',NEDGESMAX

      open(1,file=filename,status='unknown')
      weighttotal=0.
      NODOS=0
      nlink=0
      do while(.true.)
      read(1,*,END=10) i,j,d
      if(d.eq.0)then
        write(6,*)"cities ",i,j," have distance 0"
        d=d+0.1
      endif
      w=1.0/d**(1.0)
      nlink=nlink+1
      internalnet(nlink,1)=i
      internalnet(nlink,2)=j
      strenght(i)=strenght(i)+w
      strenght(j)=strenght(j)+w
      ndegree(i)=ndegree(i)+1
      ndegree(j)=ndegree(j)+1
      weight(nlink)=w
      weighttotal=weighttotal+w
      if(i.gt.NODOS) NODOS=i
      if(j.gt.NODOS) NODOS=j
      enddo
10    close(1)

      weightbackbone=0.
      open(1,file=filenameout,status='unknown')
      do i=1,nlink
      if((ndegree(internalnet(i,2)).gt.1).and.
     +   (ndegree(internalnet(i,1)).gt.1))then
        if((weight(i)/strenght(internalnet(i,1)).gt.
     +(1.-(1.-wconfidence)**(1./(ndegree(internalnet(i,1))-1)))).or.
     +     (weight(i)/strenght(internalnet(i,2)).gt.
     +(1.-(1.-wconfidence)**(1./(ndegree(internalnet(i,2))-1)))))then
           if(weight(i).gt.weightmin)then
           write(1,*) internalnet(i,1),internalnet(i,2),weight(i)
           weightbackbone=weightbackbone+weight(i)
           endif
        endif
	  else if((ndegree(internalnet(i,2)).gt.1).and.
     +        (ndegree(internalnet(i,1)).eq.1))then
	     if(
     +     (weight(i)/strenght(internalnet(i,2)).gt.
     +(1.-(1.-wconfidence)**(1./(ndegree(internalnet(i,2))-1)))))then
           if(weight(i).gt.weightmin)then
           write(1,*) internalnet(i,1),internalnet(i,2),weight(i)
           weightbackbone=weightbackbone+weight(i)
           endif
          endif
	  else if((ndegree(internalnet(i,2)).eq.1).and.
     +        (ndegree(internalnet(i,1)).gt.1))then
	     if((weight(i)/strenght(internalnet(i,1)).gt.
     +(1.-(1.-wconfidence)**(1./(ndegree(internalnet(i,1))-1)))))then
           if(weight(i).gt.weightmin)then
           write(1,*) internalnet(i,1),internalnet(i,2)
!     +,weight(i)
           weightbackbone=weightbackbone+weight(i)
           endif
          endif
	   endif
      enddo

      close(1)
      write(*,*) 'fraction of weight in backbone=',
     + weightbackbone/weighttotal

	  stop
      end
